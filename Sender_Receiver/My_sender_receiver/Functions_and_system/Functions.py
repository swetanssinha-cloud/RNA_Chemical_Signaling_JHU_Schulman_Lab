import numpy as np
import pandas as pd
from fipy import Grid2D, CellVariable, DiffusionTerm, ImplicitSourceTerm, TransientTerm


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
                            value_inside, value_outside, 
                            transition_width=10.0):
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
    c = 1.0 / transition_width
    
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


def initalize_variables(mesh, x,y, sender_center_x, receiver_center_x, receiver_center_y, node_radius, I2_init, Th2_init, I1O2_init, transition_width, D_gel, D_solution):
    sender_center_y = receiver_center_y


    I2_initial = smooth_circular_profile(x, y, receiver_center_x, receiver_center_y,
                                     node_radius, I2_init, 0.0, transition_width)

    Th2_initial = smooth_circular_profile(x, y, receiver_center_x, receiver_center_y,
                                        node_radius, Th2_init, 0.0, transition_width)

    I1O2_initial = smooth_circular_profile(x, y, sender_center_x, sender_center_y,
                                        node_radius, I1O2_init, 0.0, transition_width)

    # Create smooth diffusion coefficient profile
    # Inside nodes: D_gel (60), Outside nodes: D_solution (150)
    D_sender = smooth_circular_profile(x, y, sender_center_x, sender_center_y,
                                    node_radius, D_gel, D_solution, transition_width)

    D_receiver = smooth_circular_profile(x, y, receiver_center_x, receiver_center_y,
                                        node_radius, D_gel, D_solution, transition_width)

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


#You would then write this:
#S2, I2, Th2, S2_I2, S2_Th2, I1O2, D_S2 = initalize_variables(mesh, x,y, sender_center_x, receiver_center_x, receiver_center_y, node_radius, I2_init, Th2_init, I1O2_init, transition_width, D_gel, D_solution)

def intialize_equations(S2, D_S2, I1O2, I2, Th2, S2_I2, S2_Th2):

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
