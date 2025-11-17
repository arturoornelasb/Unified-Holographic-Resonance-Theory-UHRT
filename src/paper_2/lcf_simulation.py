import networkx as nx
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def build_layered_graph(layers):
	G = nx.Graph()
	prev_nodes = []
	for i, layer_dict in enumerate(layers):
		node = f"Layer {i+1}: {layer_dict['name']}"
		G.add_node(node, layer=i+1)
		for prev in prev_nodes:
			G.add_edge(prev, node, weight=1)
		prev_nodes.append(node)
		
		# Add dualities as sub-nodes
		for duality in layer_dict.get('dualities', []):
			dual_node = f"{node} - {duality}"
			G.add_node(dual_node)
			G.add_edge(node, dual_node, weight=0.5)
		
		# Add recursive cycles for Layer 9
		if len(prev_nodes) > 0:
			G.add_edge(prev_nodes[0], prev_nodes[-1], weight=2, type='recursive')
	
	# Initial geometric placement (Phase 2: random_geometric_graph)
	geometric_G = nx.random_geometric_graph(len(G.nodes()), 0.2, dim=3)
	for idx, node in enumerate(G.nodes()):
		# Ensure 'pos' attribute exists before assigning
		if idx < len(geometric_G.nodes()):
			G.nodes[node]['pos'] = geometric_G.nodes[idx]['pos']
		else:
			G.nodes[node]['pos'] = np.random.rand(3) # Fallback
	
	return G

def generate_triangular_geometry(G, start_node, iterations=3):
	"""Iteratively develop triangular geometry with fractal self-similarity (Phase 2: Sierpinski extension)."""
	if start_node not in G:
		return G
	
	def sierpinski_points(level, base_points):
		if level == 0:
			return base_points
		sub_points = []
		for i in range(3):
			shift = (base_points[(i+1)%3] + base_points[(i+2)%3]) / 2 - base_points[i] / 2
			# Ensure correct numpy operations
			sub = (sierpinski_points(level-1, base_points) + shift) / 2
			sub_points.append(sub)
		return np.vstack(sub_points)
	
	base_triangle = np.array([[0,0,0], [1,0,0], [0.5, np.sqrt(3)/2, 0]])
	fractal_points = sierpinski_points(iterations, base_triangle)
	
	# Add fractal nodes/edges
	for idx, point in enumerate(fractal_points):
		new_node = f"{start_node}_fractal_{idx}"
		G.add_node(new_node, pos=point)
		G.add_edge(start_node, new_node, weight=0.8)
	
	return G

def generate_tetrahedral_geometry(G, start_node, iterations=2):
	"""Generate tetrahedral volumes with recursive subdivision (Phase 2)."""
	if start_node not in G:
		return G
	
	def subdivide_tetra(level, base_points):
		if level == 0:
			return base_points
		# Calculate midpoints for all edges
		mid_points = {}
		for i in range(4):
			for j in range(i + 1, 4):
				mid_points[(i, j)] = (base_points[i] + base_points[j]) / 2
		
		# Define the 4 new tetrahedrons by their vertices
		new_base_points = [
			[base_points[0], mid_points[(0,1)], mid_points[(0,2)], mid_points[(0,3)]],
			[mid_points[(0,1)], base_points[1], mid_points[(1,2)], mid_points[(1,3)]],
			[mid_points[(0,2)], mid_points[(1,2)], base_points[2], mid_points[(2,3)]],
			[mid_points[(0,3)], mid_points[(1,3)], mid_points[(2,3)], base_points[3]]
		]
		
		all_points = []
		for new_base in new_base_points:
			all_points.append(subdivide_tetra(level - 1, np.array(new_base)))
		
		# Check if subdivide_tetra returns a list or ndarray and handle accordingly
		# Assuming it returns ndarray
		return np.vstack(all_points)

	base_tetra = np.array([[0,0,0], [1,0,0], [0.5, np.sqrt(3)/2,0], [0.5, np.sqrt(3)/6, np.sqrt(6)/3]])
	
	# Correct recursive call which should return points
	try:
		fractal_points = subdivide_tetra(iterations, base_tetra)
	except Exception as e:
		print(f"Error in subdivide_tetra: {e}")
		fractal_points = base_tetra # fallback

	# Add nodes/edges as cliques for tetrahedrons
	num_points_per_tetra = 4
	if fractal_points.ndim == 2 and fractal_points.shape[1] == 3:
		for i in range(0, len(fractal_points), num_points_per_tetra):
			if i + num_points_per_tetra <= len(fractal_points):
				tetra_nodes = [f"{start_node}_tetra_{i+j}" for j in range(num_points_per_tetra)]
				current_points = fractal_points[i:i+num_points_per_tetra]
				
				for n, p in zip(tetra_nodes, current_points):
					G.add_node(n, pos=p)
					G.add_edge(start_node, n, weight=0.7)
				
				# Create a clique (all-to-all connections) for the tetrahedron
				for j1 in range(num_points_per_tetra):
					for j2 in range(j1 + 1, num_points_per_tetra):
						G.add_edge(tetra_nodes[j1], tetra_nodes[j2], weight=1)
	
	return G

def calculate_entropy(G):
	degrees = np.array([d for n, d in G.degree()])
	if len(degrees) == 0 or degrees.sum() == 0:
		return 0
	p = degrees / degrees.sum()
	return entropy(p, base=2)

def compute_ubs(G, base=1, scale=1, Df=2.5):
	"""Compute global UBS (Phase 3: Integration with fractal term)."""
	num_nodes = G.number_of_nodes()
	H = calculate_entropy(G)
	log_term = np.log2(num_nodes / base) if num_nodes > base else 0
	fractal_term = Df * np.log2(scale) if scale > 1 else 0
	return log_term + H + fractal_term

def ternary_search_for_pruning(G, entropy_target, max_prune_fraction=0.5, eps=0.01, use_trig=True, hyperbolic=False):
	low, high = 0.0, max_prune_fraction
	
	# Check if initial graph is already valid
	if G.number_of_edges() == 0:
		return G, 0.0

	while high - low > eps:
		delta = high - low
		
		# Ensure midpoints are valid and distinct
		if use_trig:
			# Use a simpler, stable trigonometric split
			mid1 = low + delta * 0.33
			mid2 = high - delta * 0.33
		else:
			mid1 = low + delta / 3
			mid2 = high - delta / 3

		if hyperbolic:
			# This is a conceptual placeholder; real hyperbolic search is complex.
			# Using a simple split instead.
			mid1 = low + delta / 3
			mid2 = high - delta / 3
			pass

		# Ensure mid1 and mid2 do not cross
		if mid1 >= mid2:
			mid1 = low + (high-low)/4
			mid2 = high - (high-low)/4
			if mid1 >= mid2: break # Failsafe

		G1 = prune_graph(G.copy(), mid1)
		G2 = prune_graph(G.copy(), mid2)
		
		ent1 = calculate_entropy(G1)
		ent2 = calculate_entropy(G2)
		
		if abs(ent1 - entropy_target) < abs(ent2 - entropy_target):
			high = mid2
		else:
			low = mid1
			
	final_fraction = (low + high) / 2
	return prune_graph(G, final_fraction), final_fraction

def prune_graph(G, fraction):
	if fraction <= 0:
		return G
	# Sort edges by weight ascending to prune low-weight first
	edges = sorted(G.edges(data=True), key=lambda e: e[2].get('weight', 1))
	num_remove = int(len(edges) * fraction)
	
	# Iterate over a copy if removing
	edges_to_remove = edges[:num_remove]
	for u, v, _ in edges_to_remove:
		if G.has_edge(u, v):
			G.remove_edge(u, v)
	return G

# Full layers data
layers = [
	{'name': 'Initial Duality (Qubit Base)', 'dualities': ['Existence-Non-Existence']},
	{'name': 'Line and Evolution', 'dualities': ['Sum-Subtraction', 'Evolution-Stability']},
	{'name': 'Properties of the Line', 'dualities': ['Movement-Rest', 'Direction-Indirection', 'Distance-Proximity', 'Velocity-Slowness', 'Division-Multiplication']},
	{'name': 'Triangle and Bidimensional Area', 'dualities': ['Surface-Bidimensional Void', 'Angular Relations-Non-Angularity']},
	{'name': 'Tetrahedron and Tridimensional Volume', 'dualities': ['Tridimensional Space-Tridimensional Void', 'Spatial Relations-Non-Spatiality']},
	{'name': 'Volumetric Interactions and Dynamics', 'dualities': ['Dynamics-Statics', 'Interaction-Isolation', 'Energy-Inercia']},
	{'name': 'Information and Entropy (Quantum Complex Systems)', 'dualities': ['Information-Ignorance', 'Entropy-Order', 'Transmission-Blockage']},
	{'name': 'Universe Evolution (Holographic)', 'dualities': ['Expansion-Contraction', 'Cosmic Order-Disorder']},
	{'name': 'Meta-Reflection or Universal Feedback Cycle', 'dualities': ['Observer-Observed', 'Cycle-Acyclicity', 'Knowledge-Uncertainty']}
]

# Example usage (Phases 2-4 integrated)
def main():
	G = build_layered_graph(layers)
	
	# Check if nodes exist before trying to generate geometry
	tri_node = 'Layer 4: Triangle and Bidimensional Area'
	if tri_node in G:
		G = generate_triangular_geometry(G, tri_node, iterations=2)
		
	tetra_node = 'Layer 5: Tetrahedron and Tridimensional Volume'
	if tetra_node in G:
		G = generate_tetrahedral_geometry(G, tetra_node, iterations=1)
	
	ubs_before = compute_ubs(G, base=1, scale=10, Df=2.5) # Use base=1 to avoid log(0)
	
	try:
		optimized_G, frac = ternary_search_for_pruning(G.copy(), entropy_target=2.0)
		ubs_after = compute_ubs(optimized_G, base=1, scale=10, Df=2.5)
	except Exception as e:
		print(f"Error during optimization: {e}")
		optimized_G = G
		ubs_after = ubs_before
		frac = 0.0

	# Visualization (Phase 4: Improved 3D plot with labels)
	try:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		positions = nx.get_node_attributes(G, 'pos')
		
		# Filter out nodes that don't have a 'pos' (shouldn't happen with fallback)
		nodes_with_pos = {n: p for n, p in positions.items() if p is not None}
		
		# Draw edges
		for edge in G.edges():
			p1 = nodes_with_pos.get(edge[0])
			p2 = nodes_with_pos.get(edge[1])
			if p1 is not None and p2 is not None:
				ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', alpha=0.3)
		
		# Draw nodes
		node_xyz = np.array([pos for pos in nodes_with_pos.values()])
		if node_xyz.size > 0:
			ax.scatter(node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2], s=20, c='b')

		# Draw labels (optional, can be slow)
		# for node, pos in nodes_with_pos.items():
		# 	ax.text(pos[0], pos[1], pos[2], node[:10], size=8)  # Truncated labels
		
		# Set labels for clarity
		ax.set_xlabel('X Axis')
		ax.set_ylabel('Y Axis')
		ax.set_zlabel('Z Axis')
		plt.title('LCF Knowledge Graph 3D Visualization')
		
		# Save figure instead of showing
		plt.savefig("lcf_graph_3d.png")
		print("Graph visualization saved to lcf_graph_3d.png")
		plt.close(fig) # Close figure to free memory

	except Exception as e:
		print(f"Could not generate 3D plot: {e}")
		print("Skipping visualization.")

	# Testability outputs
	print(f"Number of nodes: {G.number_of_nodes()}")
	print(f"Number of edges: {G.number_of_edges()}")
	print(f"Entropy before: {calculate_entropy(G):.4f}")
	print(f"UBS before: {ubs_before:.4f}")
	print(f"Optimal pruning fraction: {frac:.4f}")
	print(f"Entropy after: {calculate_entropy(optimized_G):.4f}")
	print(f"UBS after: {ubs_after:.4f}")

if __name__ == "__main__":
	main()