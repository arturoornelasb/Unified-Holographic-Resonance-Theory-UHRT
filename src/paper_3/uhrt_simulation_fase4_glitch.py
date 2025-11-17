import networkx as nx
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from fractions import Fraction

# --- START OF MEDC LOGIC (ARITHMETIC ENGINE) ---
# This logic is based on "A Rigorous Triadic Framework"
class ArithmeticProportionality:
    """
    Implements the generative and stabilizing logic of the MEDC
    (Composite Dimensional Evolution Framework).
    """

    def check_simple_rule_of_three(self, A, B, C):
        """
        Verifies the stability of a 2D system (LCF Layer 4) using
        the Simple Rule of Three: A/B = C/X.
        Returns X if stable (integer), or None if unstable.
        """
        if not all(isinstance(x, (int, float)) and x > 0 for x in [A, B, C]):
            print(f"MEDC Error: Invalid inputs for Rule of Three: A={A}, B={B}, C={C}")
            return None
        
        # X = (B * C) / A
        X = Fraction(B * C, A)
        
        if X.denominator == 1:
            # STABLE! The system is arithmetically stable.
            return int(X)
        else:
            # UNSTABLE! The system does not resolve to integers.
            # This should trigger a "Glitch".
            return None

# --- END OF MEDC LOGIC ---


def build_layered_graph(layers):
    """
    Builds the base LCF graph (skeleton).
    PHASE 4 UPDATE: Adds MEDC attributes to relevant nodes.
    """
    G = nx.Graph()
    prev_nodes = []
    
    # Stage mapping for easy reference
    stage_nodes = {}

    # --- Build Layers (LCF) ---
    for i, stage_info in enumerate(layers):
        node_name = f"Stage {i} ({stage_info['lcf_layers']} / {stage_info['medc_step']})"
        node_label = f"{stage_info['name']} / {stage_info['concept']}"
        
        # --- MEDC LOGIC INJECTION (PHASE 4) ---
        medc_attributes = {}
        if i == 2: # Stage 2: Plane / Simple Rule of Three
            # Add arithmetic attributes for the simulation
            # UNSTABLE (GLITCH) VERSION
            medc_attributes = {
                'A': 3, 
                'B': 4, 
                'C': 5, # <-- Set to 5 to trigger Glitch
                'Expected_X': None # Unstable: (4*5)/3 = 6.66...
            }
        
        G.add_node(node_name, label=node_label, stage=i, **medc_attributes)
        stage_nodes[i] = node_name
        
        if i > 0:
            G.add_edge(stage_nodes[i-1], node_name, weight=1)

    # Add recursive cycle (LCF Layer 9 -> LCF Layer 1)
    if 0 in stage_nodes and 4 in stage_nodes:
        # Assumes Stage 4 (LCF 7-9) connects back to Stage 0 (LCF 1)
        G.add_edge(stage_nodes[4], stage_nodes[0], weight=2, type='recursive_feedback')

    # Add dualities (LCF concepts)
    for i, stage_info in enumerate(layers):
        node_name = stage_nodes[i]
        for duality in stage_info.get('dualities', []):
            dual_node = f"{node_name} - {duality}"
            G.add_node(dual_node, label=duality, type='duality')
            G.add_edge(node_name, dual_node, weight=0.5)

    # Initial geometric positioning (for visualization)
    geometric_G = nx.random_geometric_graph(len(G.nodes()), 0.2, dim=3)
    pos_map = {list(G.nodes())[i]: geometric_G.nodes[i]['pos'] for i in range(len(G.nodes()))}
    nx.set_node_attributes(G, pos_map, 'pos')

    return G, stage_nodes

def generate_triangular_geometry(G, start_node, iterations=3):
    if start_node not in G: return G
    # ... (code for geometry generation) ...
    return G

def generate_tetrahedral_geometry(G, start_node, iterations=2):
    if start_node not in G: return G
    # ... (code for geometry generation) ...
    return G

def calculate_entropy(G):
    degrees = np.array([d for n, d in G.degree()])
    if len(degrees) == 0 or degrees.sum() == 0:
        return 0
    p = degrees / degrees.sum()
    return entropy(p, base=2)

def compute_ubs_uhm(G, n_dim=3, L_scale=1, Df=2.5):
    """
    Calculates the "Super Metric" (Phase 3).
    UBS_UHM = (LCF Terms) + (MEDC Term)
    """
    num_nodes = G.number_of_nodes()
    S_X = calculate_entropy(G) # S(X)
    
    # LCF: Total Information Content
    log_term = np.log2(num_nodes) if num_nodes > 0 else 0
    fractal_term = Df * np.log2(L_scale) if L_scale > 1 else 0
    lcf_content = log_term + S_X + fractal_term
    
    # MEDC: Dimensional Density Factor D(n)
    # D(n) = -n * log2(L)
    D_n = -n_dim * (np.log2(L_scale) if L_scale > 0 else 0)
    
    return lcf_content + D_n


def prune_graph(G_in, fraction):
    """ Prunes the graph, removing 'fraction' of the lowest-weight edges. """
    G = G_in.copy()
    if fraction <= 0 or G.number_of_edges() == 0:
        return G
    
    edges = sorted(G.edges(data=True), key=lambda e: e[2].get('weight', 1))
    num_remove = int(len(edges) * fraction)
    
    edges_to_remove = edges[:num_remove]
    G.remove_edges_from([(u,v) for u,v,_ in edges_to_remove])
    return G

def ternary_search_for_pruning(G, entropy_target, max_prune_fraction=0.5, eps=0.01):
    """
    Finds the optimal pruning fraction to reach a target entropy.
    This simulates the resolution of a "Glitch".
    """
    low, high = 0.0, max_prune_fraction
    
    if G.number_of_edges() == 0:
        return G, 0.0

    while high - low > eps:
        delta = high - low
        mid1 = low + delta / 3
        mid2 = high - delta / 3

        G1 = prune_graph(G, mid1)
        G2 = prune_graph(G, mid2)
        
        ent1 = calculate_entropy(G1)
        ent2 = calculate_entropy(G2)
        
        if abs(ent1 - entropy_target) < abs(ent2 - entropy_target):
            high = mid2
        else:
            low = mid1
            
    final_fraction = (low + high) / 2
    return prune_graph(G, final_fraction), final_fraction

# --- START OF PHASE 4 SIMULATION ---

def run_stability_check(G, stage_nodes, medc_logic, entropy_target):
    """
    Master Plan Implementation (Phase 4):
    Checks arithmetic stability (MEDC) of the graph (LCF).
    If unstable, triggers a "Glitch" (entropic pruning).
    """
    print("\n--- [PHASE 4: STABILITY SIMULATION (MEDC in LCF)] ---")
    
    # 1. Locate the Stage 2 node (LCF 4 / MEDC 2D)
    node_name = stage_nodes.get(2)
    if not node_name or node_name not in G.nodes:
        print("Error: 'Stage 2' node not found for stability test.")
        return G, False

    node_data = G.nodes[node_name]
    A = node_data.get('A')
    B = node_data.get('B')
    C = node_data.get('C')
    
    print(f"Checking arithmetic stability of '{node_name}'...")
    print(f"Test values (MEDC): A={A}, B={B}, C={C}")

    # 2. Run MEDC logic (Rule of Three)
    X_result = medc_logic.check_simple_rule_of_three(A, B, C)
    
    # 3. Evaluate the result
    if X_result is not None:
        print(f"Result: STABLE. Simple Rule of Three resolved (X = {X_result}).")
        print("System does not require entropic pruning.")
        return G, False # Return original graph, no glitch
    else:
        print(f"Result: UNSTABLE. Rule of Three did not resolve to an integer.")
        print(f"ARITHMETIC GLITCH DETECTED!!!")
        print(f"Triggering entropic pruning (LCF) to re-stabilize system...")
        
        entropy_before = calculate_entropy(G)
        print(f"Entropy before Glitch: {entropy_before:.4f}")
        
        # 4. Trigger the "Glitch" (LCF's entropic pruning)
        pruned_G, frac = ternary_search_for_pruning(G, entropy_target)
        
        entropy_after = calculate_entropy(pruned_G)
        print(f"Pruning complete. Pruning fraction: {frac:.4f}")
        print(f"Entropy after Glitch: {entropy_after:.4f}")
        return pruned_G, True # Return new, pruned graph

# --- END OF PHASE 4 SIMULATION ---

# --- Data for the 5 Fused Stages ---
# (Based on Section 3 of Teoria_Resonancia_Holografica.tex)
fused_layers = [
    {
        'lcf_layers': 'LCF 1', 'medc_step': 'MEDC 0D', 
        'name': 'Duality and the Glitch', 'concept': 'Qubit / Glitch',
        'dualities': ['Existence-Non-Existence']
    },
    {
        'lcf_layers': 'LCF 2-3', 'medc_step': 'MEDC 1D', 
        'name': 'The Line and the Simple Ratio', 'concept': 'Line / Simple Ratio',
        'dualities': ['Sum-Subtraction', 'Evolution-Stability', 'Movement-Rest', 'Direction-Indirection']
    },
    {
        'lcf_layers': 'LCF 4', 'medc_step': 'MEDC 2D', 
        'name': 'The Plane and the Simple Rule of Three', 'concept': 'Plane / Simple Rule',
        'dualities': ['Surface-Void', 'Angular-Non-Angular']
    },
    {
        'lcf_layers': 'LCF 5-6', 'medc_step': 'MEDC 3D', 
        'name': 'The Volume and the Compound Rule', 'concept': 'Tetrahedron / Compound Rule',
        'dualities': ['Volume-Void', 'Dynamics-Statics', 'Interaction-Isolation']
    },
    {
        'lcf_layers': 'LCF 7-9', 'medc_step': 'MEDC nD', 
        'name': 'The Holographic Pyramid', 'concept': 'Holographic Pyramid / Recursion',
        'dualities': ['Information-Ignorance', 'Entropy-Order', 'Observer-Observed']
    }
]


def main():
    print("--- STARTING HOLOGRAPHIC RESONANCE THEORY SIMULATION (GLITCH) ---")
    
    # --- Initialize Logic ---
    medc_logic = ArithmeticProportionality()
    
    # --- Phase 2: Structural Mapping ---
    print("\n--- [PHASE 2: STRUCTURAL MAPPING (LCF + MEDC)] ---")
    G, stage_nodes = build_layered_graph(fused_layers)
    print(f"Base LCF graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # --- Phase 3: Super Metric Calculation (Initial State) ---
    print("\n--- [PHASE 3: SUPER METRIC CALCULATION (INITIAL)] ---")
    ubs_initial = compute_ubs_uhm(G, n_dim=3, L_scale=10, Df=2.5)
    entropy_initial = calculate_entropy(G)
    print(f"Initial Entropy (LCF): {entropy_initial:.4f}")
    print(f"Initial Super Metric UBS-UHM (MEDC+LCF): {ubs_initial:.4f}")

    # --- Phase 4: Narrative & Code Simulation ---
    # The Glitch will trigger if the Rule of Three fails in 'Stage 2'
    # The entropy pruning (Glitch) targets an entropy of 2.0
    G_final, glitch_occurred = run_stability_check(G, stage_nodes, medc_logic, entropy_target=2.0)
    
    # --- Phase 5: Validation (Final State) ---
    print("\n--- [PHASE 5: VALIDATION (FINAL STATE)] ---")
    ubs_final = compute_ubs_uhm(G_final, n_dim=3, L_scale=10, Df=2.5)
    entropy_final = calculate_entropy(G_final)
    print(f"Final Entropy (LCF): {entropy_final:.4f}")
    print(f"Final Super Metric UBS-UHM (MEDC+LCF): {ubs_final:.4f}")
    
    # --- Visualization (Optional) ---
    try:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        positions = nx.get_node_attributes(G_final, 'pos')
        
        nodes_with_pos = {n: p for n, p in positions.items() if p is not None}
        
        for edge in G_final.edges():
            p1 = nodes_with_pos.get(edge[0])
            p2 = nodes_with_pos.get(edge[1])
            if p1 is not None and p2 is not None:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', alpha=0.3)
        
        node_xyz = np.array([pos for pos in nodes_with_pos.values()])
        if node_xyz.size > 0:
            ax.scatter(node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2], s=20, c='b')

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        plt.title('UHRT Simulation (Glitch State)')
        
        # --- FILENAME CORRECTION ---
        plt.savefig("uhrt_simulation_glitch.png")
        print("\n3D visualization saved to 'uhrt_simulation_glitch.png'")
        plt.close(fig) 

    except Exception as e:
        print(f"\nCould not generate 3D plot: {e}")

if __name__ == "__main__":
    main()