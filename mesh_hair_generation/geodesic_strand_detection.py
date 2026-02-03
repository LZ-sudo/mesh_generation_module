"""
Geodesic-Based Hair Strand Detection

Detects hair strand paths on continuous mesh hair assets using geodesic distance
computation. This approach works for MPFB-style hair meshes that are single
continuous triangle meshes without separate UV islands.

Algorithm:
1. Extract mesh data (vertices, faces) from Blender object
2. Identify scalp vertices (top N% by Z-height)
3. Compute geodesic distance from scalp using Dijkstra on mesh edges
4. Find hair tip candidates (top N% by geodesic distance)
5. Cluster tips spatially to reduce strand count
6. Trace strand paths from tips back to scalp
7. Generate bone definitions along each strand path

Dependencies:
    - numpy: Array operations
    - scipy (optional): Spatial clustering (KDTree) - has greedy fallback

Usage:
    from geodesic_strand_detection import (
        GeodesicConfig, detect_strands_geodesic, extract_mesh_for_geodesic
    )

    vertices, faces = extract_mesh_for_geodesic(hair_obj)
    strands = detect_strands_geodesic(vertices, faces, config)
"""

import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set

# Optional scipy for KDTree clustering (has greedy fallback)
try:
    from scipy.spatial import KDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    KDTree = None


@dataclass
class GeodesicConfig:
    """Configuration for geodesic-based hair strand detection."""

    # Scalp detection (used as fallback when no external scalp reference provided)
    scalp_percentile: float = 15.0
    """Top N% of vertices by Z-height are considered scalp region (fallback mode)."""

    # Tip detection
    tip_percentile: float = 97.0
    """Top N% of vertices by geodesic distance are tip candidates."""

    tip_cluster_distance: float = 0.06
    """Meters - merge tip candidates closer than this distance."""

    # Strand filtering
    min_strand_length: float = 0.04
    """Meters - minimum strand length to generate bones for."""

    max_strands: int = 20
    """Maximum number of strand paths to generate."""

    # Direction filtering
    min_downward_component: float = -0.2
    """Minimum Z component of strand direction (negative = downward).
    Strands pointing more upward than this are rejected."""

    max_direction_variance: float = 0.5
    """Maximum allowed variance in bone directions along a strand (0-1).
    Lower values require more consistent direction."""

    # Bone generation
    bones_per_10cm: float = 3.0
    """Number of bones per 10cm of strand length."""

    min_bones_per_strand: int = 2
    """Minimum bones per strand."""

    max_bones_per_strand: int = 6
    """Maximum bones per strand."""

    # Path tracing
    path_smoothing_iterations: int = 3
    """Number of smoothing passes on traced paths."""

    require_scalp_root: bool = True
    """Require strand roots to be on scalp boundary."""

    # Vertex weighting
    weight_falloff: float = 2.0
    """Falloff exponent for vertex weight calculation."""

    # Bone containment (Issue 1: bones outside mesh)
    bone_inward_offset: float = 0.002
    """Meters - offset bone positions inward toward mesh center (2mm default)."""

    # Path overlap filtering (Issue 2: overlapping chains)
    path_overlap_distance: float = 0.015
    """Meters - paths closer than this are considered overlapping (1.5cm default)."""

    path_overlap_ratio: float = 0.5
    """Ratio of path that must be within overlap distance to trigger removal (0-1)."""


@dataclass
class HairStrand:
    """Represents a detected hair strand path."""

    path_coords: List[Tuple[float, float, float]]
    """Ordered coordinates from root (scalp) to tip."""

    path_vertex_indices: List[int]
    """Vertex indices along the path."""

    length: float
    """Total length of the strand in meters."""

    root_position: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    """Position of strand root (scalp attachment)."""

    tip_position: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    """Position of strand tip."""

    direction: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, -1.0))
    """Normalized direction from root to tip."""

    nearby_vertices: List[int] = field(default_factory=list)
    """Vertices near this strand for weight assignment."""

    @property
    def bone_count(self) -> int:
        """Calculate recommended number of bones based on length."""
        # 3 bones per 10cm
        count = max(2, int(self.length * 100 * 0.3))
        return min(count, 8)


def check_dependencies() -> Tuple[bool, str]:
    """
    Check if required dependencies are available.

    scipy is optional for tip clustering - we have a greedy fallback.

    Returns:
        Tuple of (available, message)
    """
    messages = []
    messages.append("geodesic: Dijkstra on mesh edges")

    if SCIPY_AVAILABLE:
        messages.append("scipy: available (KDTree clustering)")
    else:
        messages.append("scipy: not available (using greedy clustering)")

    # Always return True since we have fallbacks for everything
    return True, "; ".join(messages)


def extract_mesh_for_geodesic(hair_obj) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract vertex and face data from a Blender mesh object.

    Args:
        hair_obj: Blender mesh object

    Returns:
        Tuple of:
        - vertices: (N, 3) array of world-space vertex positions
        - faces: (M, 3) array of triangle face indices
        - vertex_indices: (N,) array mapping local index to original mesh index
    """
    mesh = hair_obj.data
    world_matrix = hair_obj.matrix_world

    # Extract vertices in world space
    num_verts = len(mesh.vertices)
    vertices = np.zeros((num_verts, 3), dtype=np.float64)

    for i, v in enumerate(mesh.vertices):
        world_pos = world_matrix @ v.co
        vertices[i] = [world_pos.x, world_pos.y, world_pos.z]

    # Extract faces (triangles only)
    # Count triangles first
    num_tris = sum(1 for p in mesh.polygons if len(p.vertices) == 3)
    num_quads = sum(1 for p in mesh.polygons if len(p.vertices) == 4)
    total_tris = num_tris + num_quads * 2

    faces = np.zeros((total_tris, 3), dtype=np.int32)
    face_idx = 0

    for poly in mesh.polygons:
        verts = list(poly.vertices)
        if len(verts) == 3:
            faces[face_idx] = verts
            face_idx += 1
        elif len(verts) == 4:
            # Triangulate quad
            faces[face_idx] = [verts[0], verts[1], verts[2]]
            faces[face_idx + 1] = [verts[0], verts[2], verts[3]]
            face_idx += 2
        # Skip n-gons with more than 4 vertices

    # Trim faces array if we skipped any n-gons
    faces = faces[:face_idx]

    # Vertex indices mapping (identity for now, but useful for subset operations)
    vertex_indices = np.arange(num_verts, dtype=np.int32)

    return vertices, faces, vertex_indices


def _compute_geodesic_distance(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_indices: np.ndarray,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute geodesic distance from source vertices using Dijkstra's algorithm.

    Computes shortest path distances along mesh edges, which approximates
    geodesic distance for sufficiently dense meshes.

    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) triangle indices
        source_indices: Indices of source vertices (scalp)
        verbose: Enable verbose output

    Returns:
        Tuple of:
        - (N,) array of geodesic distances, or None if computation fails
        - (M,) array of reachable vertex indices, or None if computation fails
    """
    return _compute_geodesic_distance_dijkstra(vertices, faces, source_indices, verbose)


def _build_weighted_adjacency(
    vertices: np.ndarray,
    faces: np.ndarray
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Build weighted adjacency list from mesh faces.

    Each edge is weighted by the Euclidean distance between vertices.

    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) triangle indices

    Returns:
        Dict mapping vertex index to list of (neighbor_index, distance) tuples
    """
    num_verts = len(vertices)
    adjacency: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(num_verts)}

    # Track edges we've already added to avoid duplicates
    seen_edges: Set[Tuple[int, int]] = set()

    for face in faces:
        # Add edges for each pair of vertices in the triangle
        edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]

        for v0, v1 in edges:
            # Normalize edge direction for deduplication
            edge_key = (min(v0, v1), max(v0, v1))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            # Calculate edge length (Euclidean distance)
            dist = np.linalg.norm(vertices[v0] - vertices[v1])

            # Add bidirectional edges
            adjacency[v0].append((v1, dist))
            adjacency[v1].append((v0, dist))

    return adjacency


def _compute_geodesic_distance_dijkstra(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_indices: np.ndarray,
    verbose: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute approximate geodesic distance using Dijkstra's algorithm on mesh edges.

    Computes shortest path distances along mesh edges, which approximates
    geodesic distance for sufficiently dense meshes.

    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) triangle indices
        source_indices: Indices of source vertices (scalp)
        verbose: Enable verbose output

    Returns:
        Tuple of:
        - (N,) array of distances from nearest source, or None if computation fails
        - (M,) array of reachable vertex indices, or None if computation fails
    """
    num_verts = len(vertices)

    if verbose:
        print("    Using Dijkstra on mesh edges")

    try:
        # Build weighted adjacency graph
        adjacency = _build_weighted_adjacency(vertices, faces)

        # Initialize distances to infinity
        distances = np.full(num_verts, np.inf, dtype=np.float64)

        # Priority queue: (distance, vertex_index)
        # Initialize with all source vertices at distance 0
        heap: List[Tuple[float, int]] = []
        for src_idx in source_indices:
            distances[src_idx] = 0.0
            heapq.heappush(heap, (0.0, src_idx))

        # Dijkstra's algorithm
        visited = set()

        while heap:
            dist, current = heapq.heappop(heap)

            if current in visited:
                continue
            visited.add(current)

            # Skip if we found a shorter path already
            if dist > distances[current]:
                continue

            # Relax edges to neighbors
            for neighbor, edge_dist in adjacency[current]:
                new_dist = dist + edge_dist

                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(heap, (new_dist, neighbor))

        # Identify reachable vertices (finite distance)
        reachable_mask = np.isfinite(distances)
        reachable_indices = np.where(reachable_mask)[0]
        unreachable_count = num_verts - len(reachable_indices)

        if verbose:
            print(f"    Reachable: {len(reachable_indices)}/{num_verts} vertices")
            if unreachable_count > 0:
                print(f"    Warning: {unreachable_count} vertices unreachable from scalp")

        # Replace infinity with max finite distance for unreachable vertices
        if len(reachable_indices) > 0:
            max_finite = np.max(distances[reachable_mask])
            distances[~reachable_mask] = max_finite + 1.0  # Mark as slightly beyond max
        else:
            return None, None

        return distances, reachable_indices

    except Exception as e:
        print(f"  Dijkstra distance computation failed: {e}")
        return None, None


def _find_tip_candidates(
    vertices: np.ndarray,
    geodesic_dist: np.ndarray,
    config: GeodesicConfig,
    reachable_indices: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Find hair tip candidate vertices based on geodesic distance.

    Tips are vertices with high geodesic distance from the scalp.

    Args:
        vertices: (N, 3) vertex positions
        geodesic_dist: (N,) geodesic distances from scalp
        config: Detection configuration
        reachable_indices: Optional array of reachable vertex indices to filter by

    Returns:
        Array of tip candidate vertex indices
    """
    # If reachable indices provided, only consider those for percentile calculation
    if reachable_indices is not None and len(reachable_indices) > 0:
        reachable_distances = geodesic_dist[reachable_indices]
        threshold = np.percentile(reachable_distances, config.tip_percentile)
        # Get vertices above threshold that are also reachable
        reachable_set = set(reachable_indices)
        tip_mask = geodesic_dist >= threshold
        tip_indices = np.array([i for i in np.where(tip_mask)[0] if i in reachable_set])
    else:
        # Find distance threshold for top N%
        threshold = np.percentile(geodesic_dist, config.tip_percentile)
        # Get vertices above threshold
        tip_mask = geodesic_dist >= threshold
        tip_indices = np.where(tip_mask)[0]

    return tip_indices


def _cluster_tips(
    vertices: np.ndarray,
    tip_indices: np.ndarray,
    geodesic_dist: np.ndarray,
    config: GeodesicConfig
) -> List[int]:
    """
    Cluster nearby tip candidates and select representative tips.

    Uses spatial clustering to merge tips that are close together,
    keeping the one with maximum geodesic distance in each cluster.

    Args:
        vertices: (N, 3) vertex positions
        tip_indices: Indices of tip candidate vertices
        geodesic_dist: (N,) geodesic distances
        config: Detection configuration

    Returns:
        List of representative tip vertex indices
    """
    if len(tip_indices) == 0:
        return []

    if len(tip_indices) == 1:
        return [tip_indices[0]]

    tip_positions = vertices[tip_indices]
    tip_distances = geodesic_dist[tip_indices]

    # Use simple greedy clustering if scipy not available
    if not SCIPY_AVAILABLE:
        return _cluster_tips_greedy(tip_indices, tip_positions, tip_distances, config)

    # Build KD-tree for spatial queries
    tree = KDTree(tip_positions)

    # Find clusters using distance threshold
    cluster_distance = config.tip_cluster_distance

    # Track which tips have been assigned to clusters
    assigned = set()
    representatives = []

    # Sort by geodesic distance (descending) to prioritize furthest tips
    sorted_indices = np.argsort(-tip_distances)

    for idx in sorted_indices:
        tip_idx = tip_indices[idx]

        if tip_idx in assigned:
            continue

        # Find all tips within cluster distance
        pos = tip_positions[idx]
        neighbors = tree.query_ball_point(pos, cluster_distance)

        # Mark all neighbors as assigned
        for n in neighbors:
            assigned.add(tip_indices[n])

        # This tip is the representative (has highest geodesic distance in cluster)
        representatives.append(tip_idx)

        # Stop if we have enough strands
        if len(representatives) >= config.max_strands:
            break

    return representatives


def _cluster_tips_greedy(
    tip_indices: np.ndarray,
    tip_positions: np.ndarray,
    tip_distances: np.ndarray,
    config: GeodesicConfig
) -> List[int]:
    """
    Simple greedy clustering fallback when scipy is not available.
    """
    cluster_dist_sq = config.tip_cluster_distance ** 2

    # Sort by geodesic distance (descending)
    sorted_order = np.argsort(-tip_distances)

    representatives = []
    used = set()

    for idx in sorted_order:
        if idx in used:
            continue

        tip_idx = tip_indices[idx]
        pos = tip_positions[idx]

        # Check if too close to existing representative
        too_close = False
        for rep_idx in representatives:
            rep_local_idx = np.where(tip_indices == rep_idx)[0][0]
            rep_pos = tip_positions[rep_local_idx]
            dist_sq = np.sum((pos - rep_pos) ** 2)
            if dist_sq < cluster_dist_sq:
                too_close = True
                break

        if not too_close:
            representatives.append(tip_idx)
            used.add(idx)

            # Mark nearby tips as used
            for other_idx in range(len(tip_indices)):
                if other_idx in used:
                    continue
                other_pos = tip_positions[other_idx]
                dist_sq = np.sum((pos - other_pos) ** 2)
                if dist_sq < cluster_dist_sq:
                    used.add(other_idx)

        if len(representatives) >= config.max_strands:
            break

    return representatives


def _compute_path_distance(
    path1_coords: List[Tuple[float, float, float]],
    path2_coords: List[Tuple[float, float, float]],
    overlap_threshold: float,
    sample_count: int = 10
) -> Tuple[float, float]:
    """
    Compute distance metrics between two strand paths.

    Samples points along path1 and finds minimum distance from each
    sample to any segment of path2.

    Args:
        path1_coords: First path coordinates
        path2_coords: Second path coordinates
        overlap_threshold: Distance threshold for considering points as overlapping
        sample_count: Number of points to sample along path1

    Returns:
        Tuple of (mean_distance, overlap_ratio)
        - mean_distance: Average minimum distance from path1 samples to path2
        - overlap_ratio: Fraction of path1 samples within overlap_threshold of path2
    """
    if len(path1_coords) < 2 or len(path2_coords) < 2:
        return float('inf'), 0.0

    path1 = np.array(path1_coords)
    path2 = np.array(path2_coords)

    # Sample evenly-spaced points along path1
    # Use fewer samples for short paths
    actual_samples = min(sample_count, len(path1))
    if actual_samples < 2:
        actual_samples = len(path1)

    indices = np.linspace(0, len(path1) - 1, actual_samples, dtype=int)
    sample_points = path1[indices]

    min_distances = []
    overlap_count = 0

    for sample_pt in sample_points:
        # Find minimum distance from this sample to any segment of path2
        min_dist = float('inf')

        for i in range(len(path2) - 1):
            p0 = path2[i]
            p1 = path2[i + 1]

            # Distance from point to line segment
            segment = p1 - p0
            seg_len_sq = np.dot(segment, segment)

            if seg_len_sq < 0.00001:
                dist = np.linalg.norm(sample_pt - p0)
            else:
                t = max(0.0, min(1.0, np.dot(sample_pt - p0, segment) / seg_len_sq))
                projection = p0 + t * segment
                dist = np.linalg.norm(sample_pt - projection)

            min_dist = min(min_dist, dist)

        min_distances.append(min_dist)
        if min_dist < overlap_threshold:
            overlap_count += 1

    mean_distance = np.mean(min_distances) if min_distances else float('inf')
    overlap_ratio = overlap_count / len(sample_points) if sample_points.size > 0 else 0.0

    return mean_distance, overlap_ratio


def _filter_overlapping_strands(
    strands: List[HairStrand],
    config: GeodesicConfig,
    verbose: bool = False
) -> List[HairStrand]:
    """
    Remove strands whose paths significantly overlap with other strands.

    When two strands overlap, keeps the longer one (longer strands typically
    represent more important/prominent hair sections).

    Args:
        strands: List of detected strands
        config: Configuration with overlap thresholds
        verbose: Enable verbose output

    Returns:
        Filtered list of strands with overlaps removed
    """
    if len(strands) <= 1:
        return strands

    # Sort strands by length (descending) - longer strands have priority
    sorted_strands = sorted(strands, key=lambda s: s.length, reverse=True)

    kept_strands = []
    removed_count = 0

    for strand in sorted_strands:
        is_overlapping = False

        # Check overlap with all previously-kept strands
        for kept in kept_strands:
            _, overlap_ratio = _compute_path_distance(
                strand.path_coords,
                kept.path_coords,
                config.path_overlap_distance
            )

            if overlap_ratio >= config.path_overlap_ratio:
                is_overlapping = True
                break

        if is_overlapping:
            removed_count += 1
        else:
            kept_strands.append(strand)

    if verbose and removed_count > 0:
        print(f"    Removed {removed_count} overlapping strands")

    return kept_strands


def _build_adjacency(
    faces: np.ndarray,
    num_vertices: int
) -> Dict[int, Set[int]]:
    """
    Build vertex adjacency map from faces.

    Args:
        faces: (M, 3) triangle indices
        num_vertices: Total number of vertices

    Returns:
        Dict mapping vertex index to set of adjacent vertex indices
    """
    adjacency = {i: set() for i in range(num_vertices)}

    for face in faces:
        v0, v1, v2 = face
        adjacency[v0].add(v1)
        adjacency[v0].add(v2)
        adjacency[v1].add(v0)
        adjacency[v1].add(v2)
        adjacency[v2].add(v0)
        adjacency[v2].add(v1)

    return adjacency


def _trace_path_to_scalp(
    tip_idx: int,
    vertices: np.ndarray,
    geodesic_dist: np.ndarray,
    adjacency: Dict[int, Set[int]],
    scalp_indices: Set[int]
) -> List[int]:
    """
    Trace a path from tip vertex back to scalp following geodesic gradient.

    Uses greedy descent: at each vertex, move to the neighbor with
    smallest geodesic distance until reaching the scalp.

    Args:
        tip_idx: Starting tip vertex index
        vertices: (N, 3) vertex positions
        geodesic_dist: (N,) geodesic distances
        adjacency: Vertex adjacency map
        scalp_indices: Set of scalp vertex indices

    Returns:
        List of vertex indices from tip to root (reversed for root-to-tip order)
    """
    path = [tip_idx]
    current = tip_idx
    visited = {tip_idx}

    max_iterations = len(vertices)  # Safety limit

    for _ in range(max_iterations):
        # Check if we reached scalp
        if current in scalp_indices:
            break

        # Find neighbor with minimum geodesic distance
        neighbors = adjacency.get(current, set())
        if not neighbors:
            break

        best_neighbor = None
        best_dist = geodesic_dist[current]

        for neighbor in neighbors:
            if neighbor in visited:
                continue
            neighbor_dist = geodesic_dist[neighbor]
            if neighbor_dist < best_dist:
                best_dist = neighbor_dist
                best_neighbor = neighbor

        if best_neighbor is None:
            # No unvisited neighbor with lower distance - we're stuck
            # Try to find any unvisited neighbor closer to scalp
            for neighbor in neighbors:
                if neighbor not in visited:
                    if best_neighbor is None or geodesic_dist[neighbor] < geodesic_dist[best_neighbor]:
                        best_neighbor = neighbor

            if best_neighbor is None:
                break

        path.append(best_neighbor)
        visited.add(best_neighbor)
        current = best_neighbor

    # Reverse path to go from root to tip
    return list(reversed(path))


def _smooth_path(
    path_indices: List[int],
    vertices: np.ndarray,
    iterations: int = 2
) -> List[Tuple[float, float, float]]:
    """
    Smooth path coordinates using simple averaging.

    Args:
        path_indices: Vertex indices along path
        vertices: (N, 3) vertex positions
        iterations: Number of smoothing passes

    Returns:
        Smoothed path as list of (x, y, z) coordinates
    """
    if len(path_indices) < 3:
        return [tuple(vertices[i]) for i in path_indices]

    # Get initial coordinates
    coords = np.array([vertices[i] for i in path_indices])

    for _ in range(iterations):
        smoothed = coords.copy()
        # Don't smooth endpoints
        for i in range(1, len(coords) - 1):
            smoothed[i] = (coords[i-1] + coords[i] + coords[i+1]) / 3.0
        coords = smoothed

    return [tuple(c) for c in coords]


def _offset_path_inward(
    path_coords: List[Tuple[float, float, float]],
    offset_distance: float,
    vertices: np.ndarray
) -> List[Tuple[float, float, float]]:
    """
    Offset path coordinates inward toward the mesh center.

    For each path point (except endpoints), computes the local tangent direction
    and offsets the point perpendicular to the tangent, toward the mesh centroid.
    This helps ensure bones stay within the hair mesh volume.

    Args:
        path_coords: Smoothed path coordinates (root to tip)
        offset_distance: How far to offset inward (meters)
        vertices: All mesh vertices (for computing mesh centroid)

    Returns:
        Offset path coordinates
    """
    if len(path_coords) < 3 or offset_distance <= 0:
        return path_coords

    # Compute mesh centroid as reference for "inward" direction
    mesh_centroid = np.mean(vertices, axis=0)

    coords = np.array(path_coords)
    offset_coords = coords.copy()

    # Process interior points (skip root and tip endpoints)
    for i in range(1, len(coords) - 1):
        point = coords[i]

        # Compute tangent direction from adjacent points
        tangent = coords[i + 1] - coords[i - 1]
        tangent_len = np.linalg.norm(tangent)
        if tangent_len < 0.0001:
            continue
        tangent = tangent / tangent_len

        # Vector from point toward mesh centroid
        to_centroid = mesh_centroid - point
        to_centroid_len = np.linalg.norm(to_centroid)
        if to_centroid_len < 0.0001:
            continue

        # Project to_centroid onto plane perpendicular to tangent
        # This gives us the "inward" direction perpendicular to the strand
        inward = to_centroid - np.dot(to_centroid, tangent) * tangent
        inward_len = np.linalg.norm(inward)

        if inward_len < 0.0001:
            # Point is directly on line to centroid, use arbitrary perpendicular
            continue

        inward = inward / inward_len

        # Offset the point inward
        offset_coords[i] = point + inward * offset_distance

    return [tuple(c) for c in offset_coords]


def _calculate_path_length(coords: List[Tuple[float, float, float]]) -> float:
    """Calculate total length of a path."""
    if len(coords) < 2:
        return 0.0

    length = 0.0
    for i in range(1, len(coords)):
        dx = coords[i][0] - coords[i-1][0]
        dy = coords[i][1] - coords[i-1][1]
        dz = coords[i][2] - coords[i-1][2]
        length += np.sqrt(dx*dx + dy*dy + dz*dz)

    return length


def _validate_strand_direction(
    path_coords: List[Tuple[float, float, float]],
    config: GeodesicConfig
) -> Tuple[bool, str]:
    """
    Validate that a strand has proper downward direction.

    Checks:
    1. Overall direction is downward (negative Z component)
    2. Bone segments have consistent direction

    Args:
        path_coords: Strand path coordinates (root to tip)
        config: Detection configuration

    Returns:
        Tuple of (is_valid, reason)
    """
    if len(path_coords) < 2:
        return False, "Path too short"

    # Check overall direction (root to tip)
    root = np.array(path_coords[0])
    tip = np.array(path_coords[-1])
    overall_dir = tip - root
    overall_len = np.linalg.norm(overall_dir)

    if overall_len < 0.001:
        return False, "Zero-length path"

    overall_dir = overall_dir / overall_len

    # Check if strand points downward enough
    # Z component should be negative (downward) or at most slightly upward
    if overall_dir[2] > config.min_downward_component:
        return False, f"Points upward (z={overall_dir[2]:.2f})"

    # Check direction consistency along the strand
    if len(path_coords) >= 3 and config.max_direction_variance < 1.0:
        segment_dirs = []
        for i in range(len(path_coords) - 1):
            p0 = np.array(path_coords[i])
            p1 = np.array(path_coords[i + 1])
            seg_dir = p1 - p0
            seg_len = np.linalg.norm(seg_dir)
            if seg_len > 0.001:
                segment_dirs.append(seg_dir / seg_len)

        if len(segment_dirs) >= 2:
            # Calculate variance in direction (using dot products)
            # High dot product = consistent direction
            dot_products = []
            for i in range(len(segment_dirs) - 1):
                dot = np.dot(segment_dirs[i], segment_dirs[i + 1])
                dot_products.append(dot)

            # Variance measure: 1 - mean(dot_products)
            # 0 = perfectly consistent, 1 = completely inconsistent
            mean_dot = np.mean(dot_products)
            variance = 1.0 - max(0.0, mean_dot)

            if variance > config.max_direction_variance:
                return False, f"Direction too inconsistent (var={variance:.2f})"

            # Check for upward-pointing segments
            upward_count = sum(1 for d in segment_dirs if d[2] > 0.3)
            if upward_count > len(segment_dirs) * 0.3:
                return False, f"Too many upward segments ({upward_count}/{len(segment_dirs)})"

    return True, "OK"


def _validate_scalp_root(
    path_indices: List[int],
    scalp_indices: Set[int],
    vertices: np.ndarray,
    config: GeodesicConfig
) -> Tuple[bool, str]:
    """
    Validate that strand root is properly anchored to scalp.

    Args:
        path_indices: Vertex indices along path (root first)
        scalp_indices: Set of scalp vertex indices
        vertices: Vertex positions
        config: Detection configuration

    Returns:
        Tuple of (is_valid, reason)
    """
    if not config.require_scalp_root:
        return True, "Scalp root not required"

    if len(path_indices) == 0:
        return False, "Empty path"

    root_idx = path_indices[0]

    # Check if root is directly on scalp
    if root_idx in scalp_indices:
        return True, "Root on scalp"

    # Check if root is close to any scalp vertex
    root_pos = vertices[root_idx]
    scalp_positions = vertices[list(scalp_indices)]

    if len(scalp_positions) > 0:
        distances = np.linalg.norm(scalp_positions - root_pos, axis=1)
        min_dist = np.min(distances)

        # Allow roots within 1cm of scalp
        if min_dist < 0.01:
            return True, f"Root near scalp ({min_dist*100:.1f}cm)"

    return False, "Root not on scalp"


def detect_strands_geodesic(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: Optional[GeodesicConfig] = None,
    verbose: bool = False,
    external_scalp_indices: Optional[np.ndarray] = None
) -> Optional[List[HairStrand]]:
    """
    Detect hair strand paths using geodesic distance computation.

    This is the main entry point for geodesic-based strand detection.

    Args:
        vertices: (N, 3) array of vertex positions in world space
        faces: (M, 3) array of triangle face indices
        config: Detection configuration (uses defaults if None)
        verbose: Enable verbose output
        external_scalp_indices: Optional pre-computed scalp vertex indices from
                               human mesh reference. If provided, uses these directly
                               instead of detecting scalp from hair mesh.

    Returns:
        List of HairStrand objects, or None if detection fails
    """
    # Check dependencies
    available, msg = check_dependencies()
    if not available:
        if verbose:
            print(f"  {msg}")
        return None

    if config is None:
        config = GeodesicConfig()

    num_verts = len(vertices)
    num_faces = len(faces)

    if verbose:
        print(f"  Geodesic strand detection: {num_verts} vertices, {num_faces} faces")

    # Validate mesh
    if num_verts < 10:
        if verbose:
            print("  Mesh too small for geodesic detection")
        return None

    if num_faces < 5:
        if verbose:
            print("  Too few faces for geodesic detection")
        return None

    # Step 1: Identify scalp vertices
    # Use external scalp indices if provided (from human mesh reference)
    if external_scalp_indices is not None and len(external_scalp_indices) >= 3:
        scalp_indices = external_scalp_indices
        if verbose:
            print(f"  Using external scalp reference: {len(scalp_indices)} vertices")
    else:
        # Fall back to Z-height based detection (simpler, more robust)
        z_coords = vertices[:, 2]
        z_threshold = np.percentile(z_coords, 100 - config.scalp_percentile)
        scalp_indices = np.where(z_coords >= z_threshold)[0]
        if verbose:
            print(f"  Using Z-height scalp: {len(scalp_indices)} vertices (top {config.scalp_percentile}%)")

    if len(scalp_indices) < 3:
        if verbose:
            print("  Too few scalp vertices identified")
        return None

    # Step 2: Compute geodesic distance from scalp
    if verbose:
        print("  Computing geodesic distances...")

    geodesic_dist, reachable_indices = _compute_geodesic_distance(
        vertices, faces, scalp_indices, verbose
    )
    if geodesic_dist is None or reachable_indices is None:
        if verbose:
            print("  Geodesic computation failed")
        return None

    # Check connectivity - warn if many vertices unreachable
    reachable_ratio = len(reachable_indices) / num_verts
    if reachable_ratio < 0.3:
        if verbose:
            print(f"  Warning: Only {reachable_ratio*100:.1f}% vertices reachable - mesh may be disconnected")
        # Continue anyway - we'll work with what we have

    # Create set of reachable vertices for filtering
    reachable_set = set(reachable_indices)

    max_dist = np.max(geodesic_dist[reachable_indices])
    if verbose:
        print(f"  Max geodesic distance: {max_dist*100:.1f}cm")

    # Step 3: Find tip candidates (only from reachable vertices)
    tip_candidates = _find_tip_candidates(
        vertices, geodesic_dist, config, reachable_indices
    )
    if verbose:
        print(f"  Found {len(tip_candidates)} tip candidates")

    if len(tip_candidates) < 1:
        if verbose:
            print("  No tip candidates found")
        return None

    # Step 4: Cluster tips
    representative_tips = _cluster_tips(vertices, tip_candidates, geodesic_dist, config)
    if verbose:
        print(f"  Clustered to {len(representative_tips)} representative tips")

    if len(representative_tips) < 1:
        if verbose:
            print("  No representative tips after clustering")
        return None

    # Step 5: Build adjacency for path tracing
    adjacency = _build_adjacency(faces, num_verts)
    scalp_set = set(scalp_indices)

    # Step 6: Trace paths from tips to scalp
    if verbose:
        print("  Tracing and validating strand paths...")

    strands = []
    all_paths = []
    rejected_root = 0
    rejected_direction = 0
    rejected_length = 0

    for tip_idx in representative_tips:
        # Trace path
        path_indices = _trace_path_to_scalp(
            tip_idx, vertices, geodesic_dist, adjacency, scalp_set
        )

        if len(path_indices) < 2:
            continue

        # Smooth path coordinates
        path_coords = _smooth_path(
            path_indices, vertices, config.path_smoothing_iterations
        )

        # Offset path inward to keep bones within mesh volume
        if config.bone_inward_offset > 0:
            path_coords = _offset_path_inward(
                path_coords, config.bone_inward_offset, vertices
            )

        # Calculate length
        length = _calculate_path_length(path_coords)

        if length < config.min_strand_length:
            rejected_length += 1
            continue

        # Validate scalp root anchoring
        root_valid, _ = _validate_scalp_root(
            path_indices, scalp_set, vertices, config
        )
        if not root_valid:
            rejected_root += 1
            continue

        # Validate direction (downward, consistent)
        dir_valid, _ = _validate_strand_direction(path_coords, config)
        if not dir_valid:
            rejected_direction += 1
            continue

        # Calculate direction
        root = path_coords[0]
        tip = path_coords[-1]
        dx = tip[0] - root[0]
        dy = tip[1] - root[1]
        dz = tip[2] - root[2]
        dir_len = np.sqrt(dx*dx + dy*dy + dz*dz)
        if dir_len > 0.0001:
            direction = (dx/dir_len, dy/dir_len, dz/dir_len)
        else:
            direction = (0.0, 0.0, -1.0)

        strand = HairStrand(
            path_coords=path_coords,
            path_vertex_indices=path_indices,
            length=length,
            root_position=root,
            tip_position=tip,
            direction=direction,
            nearby_vertices=path_indices.copy()
        )

        strands.append(strand)
        all_paths.append(path_indices)

    # Filter overlapping strands (Issue 2: overlapping chains)
    if strands and len(strands) > 1 and config.path_overlap_distance > 0:
        pre_filter_count = len(strands)
        strands = _filter_overlapping_strands(strands, config, verbose)
        overlap_removed = pre_filter_count - len(strands)
    else:
        overlap_removed = 0

    if verbose:
        total_rejected = rejected_length + rejected_root + rejected_direction
        print(f"  Generated {len(strands)} valid strands "
              f"(rejected {total_rejected}: {rejected_length} too short, "
              f"{rejected_root} bad root, {rejected_direction} bad direction, "
              f"{overlap_removed} overlapping)")
        if strands:
            lengths = [s.length * 100 for s in strands]
            print(f"  Strand lengths: min={min(lengths):.1f}cm, "
                  f"max={max(lengths):.1f}cm, avg={sum(lengths)/len(lengths):.1f}cm")

    return strands if strands else None


def assign_vertices_to_strands(
    vertices: np.ndarray,
    strands: List[HairStrand],
    config: Optional[GeodesicConfig] = None
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Assign all mesh vertices to nearby strands with weights.

    Each vertex is assigned to the nearest strand(s) based on
    distance to the strand path.

    Args:
        vertices: (N, 3) vertex positions
        strands: List of detected strands
        config: Configuration (for weight falloff)

    Returns:
        Dict mapping strand index to list of (vertex_index, weight) tuples
    """
    if config is None:
        config = GeodesicConfig()

    if not strands:
        return {}

    num_verts = len(vertices)

    # For each vertex, find distance to each strand
    vertex_strand_distances = np.full((num_verts, len(strands)), np.inf)

    for strand_idx, strand in enumerate(strands):
        path_coords = np.array(strand.path_coords)

        for v_idx in range(num_verts):
            v_pos = vertices[v_idx]

            # Find minimum distance to any segment of the strand path
            min_dist = np.inf
            for i in range(len(path_coords) - 1):
                p0 = path_coords[i]
                p1 = path_coords[i + 1]

                # Distance from point to line segment
                segment = p1 - p0
                seg_len_sq = np.dot(segment, segment)

                if seg_len_sq < 0.00001:
                    dist = np.linalg.norm(v_pos - p0)
                else:
                    t = max(0.0, min(1.0, np.dot(v_pos - p0, segment) / seg_len_sq))
                    projection = p0 + t * segment
                    dist = np.linalg.norm(v_pos - projection)

                min_dist = min(min_dist, dist)

            vertex_strand_distances[v_idx, strand_idx] = min_dist

    # Assign each vertex to nearest strand with weight based on distance
    strand_vertices = {i: [] for i in range(len(strands))}

    for v_idx in range(num_verts):
        distances = vertex_strand_distances[v_idx]
        nearest_strand = np.argmin(distances)
        min_dist = distances[nearest_strand]

        # Weight based on distance (closer = higher weight)
        # Using inverse distance with falloff
        if min_dist < 0.001:
            weight = 1.0
        else:
            weight = 1.0 / (1.0 + (min_dist * 10) ** config.weight_falloff)

        if weight > 0.01:  # Minimum weight threshold
            strand_vertices[nearest_strand].append((v_idx, weight))

    return strand_vertices
