#!/usr/bin/env python3
"""Debug horizontal region detection for different letters."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.text_extractor import TextExtractor

def analyze_letter_structure(letter):
    """Analyze the horizontal regions of a letter."""
    print(f"=== Analyzing Letter {letter} Structure ===")
    
    extractor = TextExtractor()
    
    # Extract letter points
    paths = extractor.text_to_paths(letter, 100)
    if paths:
        contours = extractor.extract_contour_points(paths[0], 500)
        if contours:
            data_points = contours[0]
            print(f"Letter {letter} has {len(data_points)} points")
            
            # Extract coordinates
            x_coords = [p[0] for p in data_points]
            y_coords = [p[1] for p in data_points]
            
            x_span = max(x_coords) - min(x_coords)
            y_span = max(y_coords) - min(y_coords)
            
            print(f"X span: {x_span:.1f}, Y span: {y_span:.1f}")
            
            # Count horizontal stroke density by analyzing y-distribution
            sorted_y = sorted(y_coords)
            y_clusters = []
            current_cluster = [sorted_y[0]]
            cluster_tolerance = y_span * 0.15  # 15% of total height
            
            print(f"Cluster tolerance: {cluster_tolerance:.1f}")
            
            for i, y in enumerate(sorted_y[1:]):
                gap = y - current_cluster[-1]
                if gap <= cluster_tolerance:
                    current_cluster.append(y)
                else:
                    cluster_size = len(current_cluster)
                    cluster_percentage = cluster_size / len(data_points)
                    print(f"  Cluster {len(y_clusters)+1}: {cluster_size} points ({cluster_percentage:.1%}), range {min(current_cluster):.1f}-{max(current_cluster):.1f}")
                    if cluster_percentage >= 0.05:  # At least 5% of points
                        y_clusters.append(current_cluster)
                    current_cluster = [y]
            
            # Don't forget the last cluster
            if current_cluster:
                cluster_size = len(current_cluster)
                cluster_percentage = cluster_size / len(data_points)
                print(f"  Cluster {len(y_clusters)+1}: {cluster_size} points ({cluster_percentage:.1%}), range {min(current_cluster):.1f}-{max(current_cluster):.1f}")
                if cluster_percentage >= 0.05:
                    y_clusters.append(current_cluster)
            
            num_horizontal_regions = len(y_clusters)
            print(f"Total horizontal regions: {num_horizontal_regions}")
            
            # Decision logic
            if num_horizontal_regions >= 4:
                recommended_polys = 5
            elif num_horizontal_regions >= 3:
                recommended_polys = 3
            else:
                recommended_polys = 2
                
            print(f"Recommended polynomials: {recommended_polys}")
            
            return num_horizontal_regions, recommended_polys

if __name__ == "__main__":
    for letter in ['A', 'B', 'C']:
        analyze_letter_structure(letter)
        print()
