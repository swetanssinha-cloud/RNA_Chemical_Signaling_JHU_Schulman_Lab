import numpy as np
import pandas as pd
from fipy import Grid2D, CellVariable, DiffusionTerm, ImplicitSourceTerm, TransientTerm
import gmsh



D_solution = 150.0 
D_gel = 60.0
k_p = 0.2 #1/s
k_d_ds = 3e-4 #1/s
k_d_ss = 3e-4 #1/s
k_slow = 1e5 * 1e-6 # 1/(Ms) * microMolar
k_fast = 1e6 * 1e-6 # 1/(Ms) * microMolar
 
I1O2_init = 0.1 #(in uM) - 100 nM
I2_init = 0.1 #(in uM) - 100 nM
Th2_init = 5.0 #(in uM) - 5000 nM

node_size = 50.0
node_diameter = 75
node_radius = node_diameter / 2
bath_margin = 250
distance_between = 300 # Test at large distance
total_width = 1e4 #10000 μm = 1 cm
total_height = 1e3 #1000 μm = 1



def smooth_circular_profile(x, y, center_x, center_y, radius, 
                            value_inside, value_outside):
    """
    Create smooth circular concentration/diffusion profile using hyperbolic tangent.
    
    This replaces sharp boolean masks with smooth transitions to eliminate
    divide-by-zero errors in gradient calculations.
    
    Parameters:
    -----------
    x, y : numpy arrays
        Cell center coordinates from mesh.cellCenters (μm)
    center_x, center_y : float
        Center of circular node (μm)
    radius : float
        Radius of circular node (μm)
    value_inside : float
        Value at center of node (e.g., I2_init=0.1 μM or D_gel=60.0 μm²/s)
    value_outside : float
        Value far from node (e.g., 0.0 μM or D_solution=150.0 μm²/s)
    transition_width : float
        Width of smooth transition region (μm)
        Recommended: 2-5× the finest mesh spacing
        Smaller = sharper transition (more like boolean mask)
        Larger = smoother transition (more gradual)
    
    Returns:
    --------
    profile : numpy array
        Smooth profile values at each cell center
    
    Mathematical Form:
    ------------------
    profile(r) = U + (H/2) * [tanh(c*(R - r)) + 1]
    
    where:
        r = distance from center = sqrt((x-h)² + (y-k)²)
        R = radius
        H = value_inside - value_outside (height)
        U = value_outside (baseline)
        c = 1/transition_width (steepness parameter)
    
    At r=0 (center):      profile ≈ value_inside
    At r=R (boundary):    profile ≈ (value_inside + value_outside)/2
    At r→∞ (far away):    profile ≈ value_outside
    """
    # Calculate distance from center
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Steepness parameter (larger c = sharper transition)
    c = 20 * np.arctanh(0.9)/node_radius
    
    # Height and baseline
    H = value_inside - value_outside
    U = value_outside
    
    # Hyperbolic tangent profile
    # tanh(c*(R-r)) varies from +1 at r=0 to -1 as r→∞
    profile = U + (H / 2.0) * (np.tanh(c * (radius - distance)) + 1.0)
    
    return profile


def create_adaptive_mesh_for_simulation(
    node_size=50.0,
    sender_center=None,
    receiver_center=None,
    fine_dx=5.0,
    coarse_dx=40.0,
    box_padding=200.0,
    transition_width=100.0,
    total_width=1e4,
    total_height=1e3, 
    distance_between=300
):
    """
    Create adaptive mesh for the 2D genelet simulation.
    
    Parameters:
    -----------
    node_size : float
        Size of sender/receiver nodes (μm)
    sender_center : float
        X-coordinate of sender center (μm)
    receiver_center : float
        X-coordinate of receiver center (μm)
    fine_dx : float
        Fine mesh spacing near nodes (μm)
    coarse_dx : float
        Coarse mesh spacing away from nodes (μm)
    box_padding : float
        Padding around nodes to define refinement box (μm)
    transition_width : float
        Width of transition between fine and coarse (μm)
    total_width : float
        Total domain width (μm)
    total_height : float
        Total domain height (μm)
    """
    
    # Calculate sender and receiver positions if not provided
    if sender_center is None:
        sender_center = total_width / 2 - distance_between / 2
    if receiver_center is None:
        receiver_center = total_width / 2 + distance_between / 2
    
    # Calculate bounding box that encompasses both nodes
    node_centers_x = [sender_center, receiver_center]
    node_centers_y = [total_height / 2]  # Center vertically
    
    # Refinement box boundaries in X
    x_min_nodes = min(node_centers_x) - node_size/2
    x_max_nodes = max(node_centers_x) + node_size/2
    refinement_x_min = x_min_nodes - box_padding
    refinement_x_max = x_max_nodes + box_padding
    
    # Refinement box boundaries in Y
    y_center = node_centers_y[0]
    refinement_y_min = y_center - node_size/2 - box_padding
    refinement_y_max = y_center + node_size/2 + box_padding
    
    def distance_to_box(x, y, x_min, x_max, y_min, y_max):
        """
        Calculate the minimum distance from point (x,y) to the box boundary.
        Returns 0 if inside the box, positive distance if outside.
        
        """
        # Correct way: calculate each distance component
        dx = np.maximum(np.maximum(x_min - x, 0), x - x_max)
        dy = np.maximum(np.maximum(y_min - y, 0), y - y_max)
        return np.sqrt(dx**2 + dy**2)


    def calculate_refinement_factor(x, y):
        """
        Calculate refinement factor based on distance to refinement box.
        Returns value between 0 (fine) and 1 (coarse).
        """
        dist = distance_to_box(x, y, refinement_x_min, refinement_x_max,
                              refinement_y_min, refinement_y_max)
        
        if dist < transition_width:
            # Smooth transition using tanh
            blend = 0.5 * (1 + np.tanh(
                (dist - transition_width/2) / (transition_width/10)
            ))
        else:
            blend = 1.0
        
        return blend
    
    def create_adaptive_spacing_1D(total_length, positions_other_dim,
                                    fine_dx, coarse_dx, is_x_direction=True):
        """
        Create 1D spacing array that depends on position in both dimensions.
        """
        positions = [0.0]
        current_pos = 0.0
        
        while current_pos < total_length:
            # Sample refinement at current position across other dimension
            refinement_samples = []
            sample_step = max(1, len(positions_other_dim)//20)
            
            for other_pos in positions_other_dim[::sample_step]:
                if is_x_direction:
                    x, y = current_pos, other_pos
                else:
                    x, y = other_pos, current_pos
                
                blend = calculate_refinement_factor(x, y)
                refinement_samples.append(blend)
            
            # Use minimum blend (finest mesh needed along this line)
            blend = min(refinement_samples) if refinement_samples else 1.0
            dx_local = fine_dx + (coarse_dx - fine_dx) * blend
            
            current_pos += dx_local
            if current_pos < total_length:
                positions.append(current_pos)
        
        # Ensure we end at total_length
        if positions[-1] < total_length:
            positions.append(total_length)
        
        positions = np.array(positions)
        dx_array = np.diff(positions)
        return positions, dx_array
    
    # First pass: create preliminary Y spacing
    y_positions_prelim = np.linspace(0, total_height, 100)
    
    # Create X spacing considering all Y positions
    x_positions, dx_array = create_adaptive_spacing_1D(
        total_width, y_positions_prelim,
        fine_dx, coarse_dx, is_x_direction=True
    )
    
    # Create Y spacing considering all X positions
    y_positions, dy_array = create_adaptive_spacing_1D(
        total_height, x_positions,
        fine_dx, coarse_dx, is_x_direction=False
    )
    
    # Create the mesh
    mesh = Grid2D(dx=dx_array, dy=dy_array)
    
    # Calculate refinement box dimensions
    box_width = refinement_x_max - refinement_x_min
    box_height = refinement_y_max - refinement_y_min
    
    # print(f"\nAdaptive Mesh Created:")
    # print(f"  Total cells: {mesh.numberOfCells}")
    # print(f"  X cells: {len(dx_array)}")
    # print(f"  Y cells: {len(dy_array)}")
    # print(f"  Min dx: {dx_array.min():.2f} μm")
    # print(f"  Max dx: {dx_array.max():.2f} μm")
    # print(f"  Min dy: {dy_array.min():.2f} μm")
    # print(f"  Max dy: {dy_array.max():.2f} μm")
    # print(f"\nRefinement box:")
    # print(f"  X: [{refinement_x_min:.1f}, {refinement_x_max:.1f}] μm (width: {box_width:.1f} μm)")
    # print(f"  Y: [{refinement_y_min:.1f}, {refinement_y_max:.1f}] μm (height: {box_height:.1f} μm)")
    # print(f"  Distance between nodes: {receiver_center - sender_center:.1f} μm")
    
    return mesh, sender_center, receiver_center, y_center


def initalize_variables(mesh, x,y, sender_center_x, receiver_center_x, 
                        receiver_center_y, node_radius, I2_init, Th2_init, I1O2_init, D_gel, 
                        D_solution):
    sender_center_y = receiver_center_y


    I2_initial = smooth_circular_profile(x, y, receiver_center_x, receiver_center_y,
                                     node_radius, I2_init, 0.0)

    Th2_initial = smooth_circular_profile(x, y, receiver_center_x, receiver_center_y,
                                        node_radius, Th2_init, 0.0)

    I1O2_initial = smooth_circular_profile(x, y, sender_center_x, sender_center_y,
                                        node_radius, I1O2_init, 0.0)

    # Create smooth diffusion coefficient profile
    # Inside nodes: D_gel (60), Outside nodes: D_solution (150)
    D_sender = smooth_circular_profile(x, y, sender_center_x, sender_center_y,
                                    node_radius, D_gel, D_solution)

    D_receiver = smooth_circular_profile(x, y, receiver_center_x, receiver_center_y,
                                        node_radius, D_gel, D_solution)

    # Where either node exists, use gel diffusion (take minimum)
    D_combined = np.minimum(D_sender, D_receiver)

    S2 = CellVariable(name="S2", mesh=mesh, value=0.0, hasOld=True)

    I2 = CellVariable(name="I2", mesh=mesh, value=I2_initial, hasOld=True)

    Th2 = CellVariable(name="Th2", mesh=mesh, value=Th2_initial, hasOld=True)

    S2_I2 = CellVariable(name="S2_I2", mesh=mesh, value=0.0, hasOld=True)
    S2_Th2 = CellVariable(name="S2_Th2", mesh=mesh, value=0.0, hasOld=True)

    I1O2 = CellVariable(name="I1O2", mesh=mesh, value=I1O2_initial)

    # Smooth spatially varying diffusion coefficient
    D_S2 = CellVariable(name="D_S2", mesh=mesh, value=D_combined)

    return S2, I2, Th2, S2_I2, S2_Th2, I1O2, D_S2

def initalize_variables_speedup(mesh, x,y, sender_center_x, receiver_center_x, receiver_center_y, node_radius, I2_init, Th2_init, I1O2_init, D_gel, D_solution):
    sender_center_y = receiver_center_y

    sender_mask = (np.sqrt((x - sender_center_x)**2 + (y - sender_center_y)**2) <= node_radius)
    receiver_mask = (np.sqrt((x - receiver_center_x)**2 + (y - receiver_center_y)**2) <= node_radius)



    S2 = CellVariable(name="S2", mesh=mesh, value=0.0, hasOld=True)

    I2 = CellVariable(name="I2", mesh=mesh, value=I2_init, hasOld=True)
    I2.setValue(I2_init * receiver_mask)

    Th2 = CellVariable(name="Th2", mesh=mesh, value=Th2_init, hasOld=True)
    Th2.setValue(Th2_init * receiver_mask)

    S2_I2 = CellVariable(name="S2_I2", mesh=mesh, value=0.0, hasOld=True)
    S2_Th2 = CellVariable(name="S2_Th2", mesh=mesh, value=0.0, hasOld=True)

    I1O2 = CellVariable(name="I1O2", mesh=mesh, value=I1O2_init)
    I1O2.setValue(I1O2_init * sender_mask)

    # Smooth spatially varying diffusion coefficient
    D_S2 = CellVariable(name="D_S2", mesh=mesh, value=D_solution)
    D_S2.setValue(D_gel * (sender_mask | receiver_mask) + D_solution * (~(sender_mask | receiver_mask)))

    return S2, I2, Th2, S2_I2, S2_Th2, I1O2, D_S2


#You would then write this:
#S2, I2, Th2, S2_I2, S2_Th2, I1O2, D_S2 = initalize_variables(mesh, x,y, sender_center_x, receiver_center_x, receiver_center_y, node_radius, I2_init, Th2_init, I1O2_init, D_gel, D_solution)

def intialize_equations(S2, D_S2, I1O2, I2, Th2, S2_I2, S2_Th2, k_p, k_slow, k_fast, k_d_ss, k_d_ds):

    eq_S2 = (TransientTerm(var=S2) == 
            DiffusionTerm(coeff=D_S2, var=S2) +  
            k_p * I1O2 +
            ImplicitSourceTerm(coeff=-(k_slow * I2 + k_fast * Th2 + k_d_ss), var=S2))

    eq_I2 = (TransientTerm(var=I2) == 
            k_d_ds * S2_I2 +
            ImplicitSourceTerm(coeff=-k_slow * S2, var=I2))

    eq_Th2 = (TransientTerm(var=Th2) == 
            k_d_ds * S2_Th2 +
            ImplicitSourceTerm(coeff=-k_fast * S2, var=Th2))

    eq_S2_I2 = (TransientTerm(var=S2_I2) == 
                k_slow * I2 * S2 +
                ImplicitSourceTerm(coeff=-k_d_ds, var=S2_I2))

    eq_S2_Th2 = (TransientTerm(var=S2_Th2) == 
                k_fast * Th2 * S2 +
                ImplicitSourceTerm(coeff=-k_d_ds, var=S2_Th2))

    eq = eq_S2 & eq_I2 & eq_Th2 & eq_S2_I2 & eq_S2_Th2
    return eq 



def calculate_total_amount(profile, mesh):
    """Calculate total amount by integrating concentration × cell volume"""
    cell_volumes = mesh.cellVolumes  # In FiPy, automatically accounts for 2D area
    total = np.sum(profile * cell_volumes)
    return total


def create_gmsh_radial_mesh(
    bath_width=10000.0,           # μm (1 cm)
    bath_height=1000.0,           # μm (1 mm)
    node_diameter=75.0,           # μm
    distance_between_nodes=300.0, # μm (center-to-center)
    min_cell_size=0.75,          # μm (finest mesh at node surface)
    max_cell_size=50.0,          # μm (coarsest mesh far from nodes)
    growth_rate=1.5,             # How fast mesh grows with distance
    mesh_filename='radial_mesh.msh',
    visualize_gmsh=False,        # Show Gmsh GUI
    verbose=True
):
    """
    Create TRUE radial adaptive mesh using Gmsh.
    Fine mesh ONLY near circular nodes (radially), coarse everywhere else.
    
    FIXED: Saves mesh in Gmsh format 2.2 for FiPy compatibility.
    
    Parameters:
    -----------
    bath_width, bath_height : float
        Domain dimensions in μm
    node_diameter : float
        Diameter of circular hydrogel nodes in μm
    distance_between_nodes : float
        Center-to-center distance between sender and receiver nodes in μm
    min_cell_size : float
        Finest mesh size at node surfaces in μm (must be 0.75 for tanh compatibility)
    max_cell_size : float
        Coarsest mesh size far from nodes in μm
    growth_rate : float
        How quickly mesh transitions from fine to coarse (1.3 = slow, 2.0 = fast)
    mesh_filename : str
        Output filename for mesh file
    visualize_gmsh : bool
        If True, opens Gmsh GUI to visualize mesh
    verbose : bool
        Print mesh statistics
    
    Returns:
    --------
    mesh : FiPy Gmsh2D mesh object
    sender_center_x : float
    receiver_center_x : float
    y_center : float
    node_radius : float
    """
    
    node_radius = node_diameter / 2.0
    y_center = bath_height / 2.0
    
    # Position nodes symmetrically around domain center
    domain_center_x = bath_width / 2.0
    sender_center_x = domain_center_x - distance_between_nodes / 2.0
    receiver_center_x = domain_center_x + distance_between_nodes / 2.0
    
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    
    # Create new model
    gmsh.model.add("radial_adaptive_mesh")
    
    # =============================================================================
    # GEOMETRY CREATION
    # =============================================================================
    
    # Create the rectangular bath domain
    rectangle_tag = gmsh.model.occ.addRectangle(0, 0, 0, bath_width, bath_height)
    
    # Create circular sender node
    sender_circle_tag = gmsh.model.occ.addDisk(sender_center_x, y_center, 0, 
                                                node_radius, node_radius)
    
    # Create circular receiver node
    receiver_circle_tag = gmsh.model.occ.addDisk(receiver_center_x, y_center, 0,
                                                  node_radius, node_radius)
    
    # Synchronize CAD entities before mesh operations
    gmsh.model.occ.synchronize()
    
# =============================================================================
# MESH SIZE FIELD - TRUE RADIAL REFINEMENT + FINE INSIDE NODES
# =============================================================================

    # Get boundaries (curves) of the circular nodes
    sender_boundary = gmsh.model.getBoundary([(2, sender_circle_tag)], 
                                            oriented=False, combined=False, recursive=False)
    receiver_boundary = gmsh.model.getBoundary([(2, receiver_circle_tag)], 
                                            oriented=False, combined=False, recursive=False)

    # Extract curve tags (boundaries are 1D entities)
    sender_curve_tags = [abs(tag) for dim, tag in sender_boundary]
    receiver_curve_tags = [abs(tag) for dim, tag in receiver_boundary]

    if verbose:
        print(f"\nSender boundary curves: {sender_curve_tags}")
        print(f"Receiver boundary curves: {receiver_curve_tags}")

    # CREATE EXPLICIT CENTER POINTS for distance fields
    sender_center_point = gmsh.model.occ.addPoint(sender_center_x, y_center, 0)
    receiver_center_point = gmsh.model.occ.addPoint(receiver_center_x, y_center, 0)

    # Synchronize to register the new points
    gmsh.model.occ.synchronize()

    if verbose:
        print(f"Created sender center point: {sender_center_point}")
        print(f"Created receiver center point: {receiver_center_point}")

    # Distance field from SENDER node boundary (edge)
    distance_field_sender_boundary = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field_sender_boundary, "CurvesList", sender_curve_tags)
    gmsh.model.mesh.field.setNumber(distance_field_sender_boundary, "Sampling", 200)

    # Distance field from RECEIVER node boundary (edge)
    distance_field_receiver_boundary = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field_receiver_boundary, "CurvesList", receiver_curve_tags)
    gmsh.model.mesh.field.setNumber(distance_field_receiver_boundary, "Sampling", 200)

    # Distance field from SENDER CENTER POINT
    distance_field_sender_center = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field_sender_center, "PointsList", [sender_center_point])
    gmsh.model.mesh.field.setNumber(distance_field_sender_center, "Sampling", 200)

    # Distance field from RECEIVER CENTER POINT
    distance_field_receiver_center = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field_receiver_center, "PointsList", [receiver_center_point])
    gmsh.model.mesh.field.setNumber(distance_field_receiver_center, "Sampling", 200)

# Take MINIMUM distance to any boundary OR center
    min_distance_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_distance_field, "FieldsList", 
                                    [distance_field_sender_boundary, 
                                    distance_field_receiver_boundary,
                                    distance_field_sender_center,
                                    distance_field_receiver_center])

    # =============================================================================
    # CORRECTED: Use Threshold field for gradual refinement (matches reference image)
    # =============================================================================
    # Define refinement zone: fine mesh extends this far from nodes
    refinement_radius = 200.0  # μm - based on your reference image

    # Create Threshold field for smooth transition
    # Keeps mesh fine out to refinement_radius, then gradually coarsens
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "InField", min_distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", min_cell_size)      # 0.75 μm at nodes
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", max_cell_size)      # 50 μm far away
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.0)                # Fine mesh starts at node
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", refinement_radius)  # Fine mesh ends here
    gmsh.model.mesh.field.setNumber(threshold_field, "Sigmoid", 1)                  # Smooth transition

    if verbose:
        print(f"\n✓ CORRECTED mesh refinement strategy:")
        print(f"  At node surface (d=0): {min_cell_size:.3f} μm")
        print(f"  From d=0 to d={refinement_radius:.0f} μm: stays ~{min_cell_size:.3f}-{min_cell_size*2:.1f} μm")
        print(f"  Beyond d={refinement_radius:.0f} μm: smoothly grows to {max_cell_size:.1f} μm")
        print(f"  Transition: Sigmoid (smooth, gradual)")
        print(f"  → This matches your COMSOL reference image!")
        
    # Set this as the background mesh field
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
    
    # Disable other mesh size determination methods
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    
    # Algorithm options
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    
    # =============================================================================
    # GENERATE MESH
    # =============================================================================
    
    
    # Generate 2D mesh
    gmsh.model.mesh.generate(2)
    
    # Optional: optimize mesh quality

    gmsh.model.mesh.optimize("Netgen")
    
    # =============================================================================
    # STATISTICS
    # =============================================================================
    
    if verbose:
        # Get mesh statistics
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        element_types, element_tags_list, element_node_tags = gmsh.model.mesh.getElements(dim=2)
        
        n_nodes = len(node_tags)
        n_triangles = sum(len(tags) for tags in element_tags_list)
    
    # =============================================================================
    # SAVE MESH IN FORMAT 2.2 (FIPY COMPATIBLE)
    # =============================================================================
    
    # THIS IS THE FIX: Force Gmsh to save in format 2.2
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    
    if verbose:
        print(f"Saving mesh in Gmsh format 2.2 (FiPy compatible)...")
    
    gmsh.write(mesh_filename)
    
    # Optional: Launch Gmsh GUI to visualize
    if visualize_gmsh:
        if verbose:
            print("Launching Gmsh GUI...")
        gmsh.fltk.run()
    
    # Finalize Gmsh
    gmsh.finalize()
    
    # =============================================================================
    # IMPORT INTO FIPY
    # =============================================================================
    
    if verbose:
        print(f"Importing mesh into FiPy...")
    
    from fipy import Gmsh2D
    mesh = Gmsh2D(mesh_filename)
    
    if verbose:
        print(f"✓ FiPy mesh created with {mesh.numberOfCells:,} cells\n")
    
    return mesh, sender_center_x, receiver_center_x, y_center