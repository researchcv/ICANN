#!/usr/bin/env python3
"""
Main Entry Point
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Import all modules
from modules.utils import ConfigLoader, setup_logger, FileManager
from modules.object_detection import YOLODetector, GaussianGuidedYOLO
from modules.rendering import GaussianRendererWrapper, HighlightRenderer
from modules.projection import BBoxProjector, Object3DReconstructor
from modules.projection.gaussian_object_descriptor import (
    GaussianObjectDescriptorBuilder, GODDescriptor,
)
from modules.visualization import ComparisonVisualizer, ReportGenerator
from modules.scene_understanding import SpatialAnalyzer, LLMInterface, SceneGraph
from modules.scene_understanding.evidence_generator import EvidenceGenerator

# Setup logger
logger = setup_logger(log_file="output/scene_understanding.log", level="INFO")


class SceneUnderstandingPipeline:
    """Scene Understanding Pipeline class"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize pipeline
        
        Args:
            config_path: Configuration file path
        """
        logger.info("Starting initialization...")
        
        # Load configuration
        self.config = ConfigLoader(config_path)
        logger.info(f"Configuration file loaded: {config_path}")
        
        # Initialize file manager
        scene_name = Path(self.config.get('paths.source_path', 'scene')).name
        self.file_manager = FileManager(
            self.config.get('paths.output_root'),
            scene_name
        )
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Initialization complete!\n")
    
    def _initialize_components(self):
        """Initialize all components"""
        # 1. YOLO detector
        yolo_config = self.config.get_yolo_config()
        self.yolo_detector = YOLODetector(
            model_name=yolo_config.get('model_name', 'yolov8x.pt'),
            conf_threshold=yolo_config.get('conf_threshold', 0.25),
            iou_threshold=yolo_config.get('iou_threshold', 0.45),
            device=yolo_config.get('device', 'cuda'),
            classes=yolo_config.get('classes')
        )
        
        # 2. Gaussian renderer
        gaussian_config = self.config.get_gaussian_config()
        self.gaussian_renderer = GaussianRendererWrapper(
            model_path=self.config.get('paths.model_path'),
            source_path=self.config.get('paths.source_path'),
            sh_degree=gaussian_config.get('sh_degree', 3),
            load_iteration=gaussian_config.get('load_iteration', -1),
            white_background=gaussian_config.get('white_background', False)
        )
        
        # 3. Projector
        proj_config = self.config.get_projection_config()
        self.bbox_projector = BBoxProjector(
            self.gaussian_renderer,
            depth_sample_step=proj_config.get('depth_sample_step', 10),
            min_visible_points=proj_config.get('min_visible_points', 4),
            visibility_threshold=proj_config.get('visibility_threshold', 0.3)
        )
        
        # 4. 3D reconstructor
        self.object_reconstructor = Object3DReconstructor(
            self.gaussian_renderer,
            min_views=2,
            clustering_eps=1.0
        )
        
        # 5. Visualizer
        vis_config = self.config.get_visualization_config()
        self.visualizer = ComparisonVisualizer(
            dpi=vis_config.get('dpi', 150)
        )
        
        # 6. Report generator
        self.report_generator = ReportGenerator(
            self.file_manager.get_dir('report')
        )
        
        # 7. Spatial analyzer
        self.spatial_analyzer = SpatialAnalyzer(
            distance_threshold=2.0,
            near_threshold=1.0
        )
        
        # 8. LLM interface (optional)
        llm_config = self.config.get_llm_config()
        if llm_config.get('enable_llm', False):
            self.llm_interface = LLMInterface(
                provider=llm_config.get('provider', 'openai'),
                model=llm_config.get('model', 'gpt-4-turbo'),
                api_key=llm_config.get('api_key'),
                temperature=llm_config.get('temperature', 0.7),
                max_tokens=llm_config.get('max_tokens', 2000)
            )
        else:
            logger.info("LLM functionality not enabled")
            self.llm_interface = None

        # 9. Gaussian-Guided YOLO detector
        guided_cfg = self.config.get('gaussian_guided', {})
        self.guided_yolo = GaussianGuidedYOLO(
            yolo_detector=self.yolo_detector,
            renderer=self.gaussian_renderer,
            proposal_eps=guided_cfg.get('proposal_eps', 0.5),
            proposal_min_samples=guided_cfg.get('proposal_min_samples', 50),
            opacity_floor=guided_cfg.get('opacity_floor', 0.1),
            roi_conf_threshold=guided_cfg.get('roi_conf_threshold', 0.15),
            bg_conf_threshold=guided_cfg.get('bg_conf_threshold', 0.40),
            min_supporting_views=guided_cfg.get('min_supporting_views', 2),
        )

        # 10. GOD builder (lazy: needs gaussians loaded)
        self.god_builder = GaussianObjectDescriptorBuilder(
            self.gaussian_renderer.gaussians
        )

        # 11. Evidence generator
        ev_cfg = self.config.get('evidence', {})
        self.evidence_generator = EvidenceGenerator(
            god_builder=self.god_builder,
            weight_geometry=ev_cfg.get('weight_geometry', 0.40),
            weight_visual=ev_cfg.get('weight_visual', 0.35),
            weight_semantic=ev_cfg.get('weight_semantic', 0.25),
            confidence_floor=ev_cfg.get('confidence_floor', 0.35),
        )
    
    def run(self):
        """Run complete pipeline"""
        try:
            logger.info("Starting execution")

            # Step 1: Gaussian-Guided YOLO detection (Phase 1)
            detection_results, verified_objects = self.step1_guided_detection()

            # Step 2: Gaussian rendering
            self.step2_gaussian_rendering()

            # Step 3: Detection box projection
            projection_metrics = self.step3_projection()

            # Step 4: 3D object reconstruction + GOD descriptors (Phase 2)
            objects_3d, god_map = self.step4_reconstruction_and_god(detection_results)

            # Step 5: Evidence-chain scene graph + dialogue (Phase 3 & 4)
            scene_graph = self.step5_evidence_scene_understanding(
                objects_3d, god_map, detection_results
            )

            # Step 6: Generate report
            self.step6_generate_report(
                detection_results,
                projection_metrics,
                objects_3d,
                scene_graph
            )

            logger.info("Complete!")
            logger.info(f"Results saved: {self.file_manager.scene_dir}")

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise
    
    # ------------------------------------------------------------------
    # Step 1: Gaussian-Guided YOLO Detection (Phase 1)
    # ------------------------------------------------------------------

    def step1_guided_detection(self):
        """Gaussian-guided multi-view detection with 3D proposal priors."""
        logger.info("[Step 1/6] Gaussian-Guided YOLO detection")

        train_cameras = self.gaussian_renderer.get_train_cameras()
        image_paths = self._collect_image_paths(train_cameras)

        detection_results, verified_objects = self.guided_yolo.detect_all_views(
            image_paths, train_cameras
        )

        # visualise per-view detections
        for idx, det_result in enumerate(detection_results):
            if idx < len(image_paths):
                vis_path = self.file_manager.get_path(
                    'yolo_vis', f'{idx:05d}_detected.png'
                )
                self.yolo_detector.visualize_detection(
                    image_paths[idx], det_result, output_path=vis_path
                )

        # persist
        all_dets = [r.to_dict() for r in detection_results]
        self.file_manager.save_json(all_dets, 'yolo', 'detections.json')

        stats = self.yolo_detector.get_summary_statistics(detection_results)
        self.file_manager.save_json(stats, 'yolo', 'statistics.json')

        logger.info(
            f"  {stats['total_detections']} detections, "
            f"{len(verified_objects)} verified 3D objects"
        )
        return detection_results, verified_objects
    
    def step2_gaussian_rendering(self):
        """Step 2: Gaussian rendering"""
        logger.info("[Step 2/6] Gaussian scene rendering")
        
        # Render training views
        if not self.config.get('experiment.skip_train_views', False):
            train_output_dir = self.file_manager.get_dir('rendered_train')
            logger.info(f"  Rendering training views to: {train_output_dir}")
            self.gaussian_renderer.render_train_views(
                output_dir=str(train_output_dir)
            )
            # Check if saved successfully
            saved_files = list(train_output_dir.glob('*.png'))
            logger.info(f"  Saved {len(saved_files)} training view images")
        
        # Render test views
        if not self.config.get('experiment.skip_test_views', False):
            test_output_dir = self.file_manager.get_dir('rendered_test')
            logger.info(f"  Rendering test views to: {test_output_dir}")
            self.gaussian_renderer.render_test_views(
                output_dir=str(test_output_dir)
            )
            # Check if saved successfully
            saved_files = list(test_output_dir.glob('*.png'))
            logger.info(f"  Saved {len(saved_files)} test view images")
        
        logger.info("  Rendering complete!")
    
    def step3_projection(self):
        """Step 3: Detection box projection and visualization"""
        logger.info("[Step 3/6] Cross-view detection box projection")
        
        train_cameras = self.gaussian_renderer.get_train_cameras()
        
        # Load detection results
        detections_data = self.file_manager.load_json('yolo', 'detections.json')
        
        projection_metrics = {
            'mean_iou': 0.0,
            'mean_visibility': 0.0,
            'success_rate': 0.0,
            'all_projections': []
        }
        
        logger.info(f"  Generating projection visualization...")
        
        # Generate projection visualization for each view
        for idx, camera in enumerate(train_cameras):
            # Get detection results for this view
            if idx >= len(detections_data):
                continue
                
            detections = detections_data[idx]['detections']
            
            # 1. Read original image
            source_path = Path(self.config.get('paths.source_path'))
            images_dir = source_path / "images"
            if hasattr(camera, 'image_name'):
                orig_image_path = images_dir / camera.image_name
            else:
                image_files = sorted(images_dir.glob('*'))
                if idx < len(image_files):
                    orig_image_path = image_files[idx]
                else:
                    continue
            
            import cv2
            original_img = cv2.imread(str(orig_image_path))
            if original_img is None:
                continue
            
            # 2. Read YOLO detection visualization
            yolo_vis_path = self.file_manager.get_path('yolo_vis', f'{idx:05d}_detected.png')
            yolo_img = cv2.imread(str(yolo_vis_path))
            
            # 3. Read Gaussian rendered image
            rendered_path = self.file_manager.get_path('rendered_train', f'{idx:05d}_render.png')
            rendered_img = cv2.imread(str(rendered_path))
            if rendered_img is None:
                continue
            
            # 4. Draw detection boxes on rendered image (projection)
            projected_img = rendered_img.copy()
            
            for det in detections:
                bbox = det['bbox']
                label = f"{det['class_name']} {det['confidence']:.2f}"
                
                # Draw directly on same view (because it's training view)
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw green detection box
                cv2.rectangle(projected_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    projected_img,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w, y1),
                    (0, 255, 0),
                    -1
                )
                cv2.putText(
                    projected_img,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
            
            # 5. Save projection result
            projected_output_path = self.file_manager.get_path(
                'projected',
                f'{idx:05d}_projected.png'
            )
            cv2.imwrite(str(projected_output_path), projected_img)
            
            # 6. Generate four-panel comparison
            if yolo_img is not None:
                comparison = self.visualizer.create_four_panel_comparison(
                    original_img,
                    yolo_img,
                    rendered_img,
                    projected_img,
                    output_path=str(self.file_manager.get_path(
                        'projected_comparison',
                        f'{idx:05d}_comparison.png'
                    ))
                )
                logger.info(f"  View {idx}: generated comparison image")
        
        # Calculate projection metrics (simplified)
        projection_metrics['success_rate'] = 1.0  # Same-view projection always succeeds
        projection_metrics['mean_visibility'] = 1.0
        
        logger.info(f"  Projection visualization complete!")
        
        # Save projection metrics
        self.file_manager.save_json(
            projection_metrics,
            'projected',
            'projection_quality.json'
        )
        
        return projection_metrics
    
    # ------------------------------------------------------------------
    # Step 4: 3D Reconstruction + GOD Descriptors (Phase 2)
    # ------------------------------------------------------------------

    def step4_reconstruction_and_god(self, detection_results):
        """Reconstruct 3D objects and build GOD descriptors."""
        logger.info("[Step 4/6] 3D reconstruction + GOD descriptors")

        train_cameras = self.gaussian_renderer.get_train_cameras()

        objects_3d = self.object_reconstructor.reconstruct_objects_3d(
            detection_results, train_cameras
        )
        logger.info(f"  Reconstructed {len(objects_3d)} 3D objects")

        # build GOD descriptors for objects that have gaussian indices
        god_list = self.god_builder.build_batch(objects_3d)
        god_map = {g.object_id: g for g in god_list}

        # persist
        objects_dict = [obj.to_dict() for obj in objects_3d]
        self.file_manager.save_json(
            objects_dict, 'scene_understanding', 'object_database.json'
        )

        god_dicts = [g.to_dict() for g in god_list]
        self.file_manager.save_json(
            god_dicts, 'scene_understanding', 'god_descriptors.json'
        )

        logger.info(f"  Built {len(god_map)} GOD descriptors")
        return objects_3d, god_map
    
    # ------------------------------------------------------------------
    # Step 5: Evidence-Chain Scene Graph + Dialogue (Phase 3 & 4)
    # ------------------------------------------------------------------

    def step5_evidence_scene_understanding(
        self, objects_3d, god_map, detection_results
    ):
        """Build evidence-chain scene graph and generate grounded description."""
        logger.info("[Step 5/6] Evidence-chain scene understanding")

        scene_graph = self.evidence_generator.build_evidence_scene_graph(
            objects_3d=objects_3d,
            god_map=god_map,
            detection_results=detection_results,
            llm_client=self.llm_interface,
        )

        scene_graph_dict = scene_graph.to_dict()
        self.file_manager.save_json(
            scene_graph_dict, 'scene_understanding', 'scene_graph.json'
        )

        # GOD text summaries for LLM context
        god_texts = [god_map[oid].to_text() for oid in sorted(god_map)]

        if self.llm_interface:
            logger.info("  Generating evidence-grounded scene description...")
            description = self.llm_interface.generate_scene_description(
                scene_graph_dict, god_texts=god_texts
            )
            desc_path = self.file_manager.get_path(
                'scene_understanding', 'scene_description.txt'
            )
            with open(desc_path, 'w', encoding='utf-8') as f:
                f.write(description)
            logger.info(f"  Scene description saved: {desc_path}")

        logger.info("  Scene analysis complete!")
        return scene_graph

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_image_paths(self, cameras) -> list:
        """Resolve image file paths for each camera in order."""
        source_path = Path(self.config.get('paths.source_path'))
        images_dir = source_path / "images"
        paths = []

        image_files = None
        for idx, camera in enumerate(cameras):
            if hasattr(camera, 'image_name'):
                paths.append(str(images_dir / camera.image_name))
            elif hasattr(camera, 'image_path'):
                paths.append(camera.image_path)
            else:
                if image_files is None:
                    image_files = sorted(images_dir.glob('*'))
                if idx < len(image_files):
                    paths.append(str(image_files[idx]))
                else:
                    logger.warning(f"Cannot resolve image path for camera {idx}")
                    paths.append("")

        return paths

    def step6_generate_report(
        self,
        detection_results,
        projection_metrics,
        objects_3d,
        scene_graph
    ):
        """Step 6: Generate comprehensive report"""
        logger.info("[Step 6/6] Generating comprehensive report")
        
        # Prepare report data
        summary = {
            'total_images': len(detection_results),
            'total_detections': sum(len(res.detections) for res in detection_results),
            'num_3d_objects': len(objects_3d),
            'num_classes': len(set(obj.class_name for obj in objects_3d))
        }
        
        detection_stats = self.yolo_detector.get_summary_statistics(detection_results)
        detection_stats['avg_per_image'] = detection_stats['avg_detections_per_image']
        
        objects_dict = [obj.to_dict() for obj in objects_3d]
        
        # Generate HTML report
        report_path = self.report_generator.generate_html_report(
            scene_name=self.file_manager.scene_name,
            summary=summary,
            detection_stats=detection_stats,
            projection_metrics=projection_metrics,
            objects_3d=objects_dict
        )
        
        # Create README
        self.file_manager.create_readme({
            'Total detections': summary['total_detections'],
            '3D objects': summary['num_3d_objects'],
            'Classes': summary['num_classes']
        })
        
        logger.info(f"  Report generated: {report_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="LLM-Enhanced 3D Gaussian Scene Understanding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python main.py --config config/config.yaml
  python main.py --source data/scene --model output/model
"""
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Configuration file path (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        help='Source dataset path (overrides config file setting)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Trained model path (overrides config file setting)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory (overrides config file setting)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        pipeline = SceneUnderstandingPipeline(args.config)

        if args.source:
            pipeline.config.update_config('paths.source_path', args.source)
        if args.model:
            pipeline.config.update_config('paths.model_path', args.model)
        if args.output:
            pipeline.config.update_config('paths.output_root', args.output)

        pipeline.run()

        # interactive dialogue loop (optional, only when LLM is enabled)
        if pipeline.llm_interface and pipeline.llm_interface.client:
            _interactive_dialogue(pipeline)

        logger.info("All tasks complete!")
        return 0

    except KeyboardInterrupt:
        logger.warning("User interrupted execution")
        return 1
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        return 1


def _interactive_dialogue(pipeline: SceneUnderstandingPipeline):
    """Optional REPL for evidence-grounded 3D scene dialogue."""
    scene_graph_dict = pipeline.file_manager.load_json(
        'scene_understanding', 'scene_graph.json'
    )
    if scene_graph_dict is None:
        return

    god_data = pipeline.file_manager.load_json(
        'scene_understanding', 'god_descriptors.json'
    )
    god_texts = None
    if god_data:
        god_texts = [
            f"[ID {g['object_id']}] {g['class_name']}: "
            f"centroid {g['centroid']}, extent {g['extent']}, "
            f"shape {g['shape_type']}, color {g['dominant_color']}"
            for g in god_data
        ]

    pipeline.llm_interface.set_scene_context(scene_graph_dict, god_texts)

    print("\n--- 3D Scene Dialogue (type 'quit' to exit) ---")
    while True:
        try:
            query = input("You: ").strip()
        except EOFError:
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break

        result = pipeline.llm_interface.answer_query(
            scene_graph_dict, query, god_texts
        )
        print(f"Assistant: {result.get('answer', '')}")
        if result.get('uncertainty'):
            print(f"  (uncertainty: {result['uncertainty']})")
        if result.get('evidence_cited'):
            for ev in result['evidence_cited']:
                print(f"  [evidence] {ev}")
        print()


if __name__ == "__main__":
    sys.exit(main())

