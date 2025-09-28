
import os
from train import train_model
import json


if __name__ == "__main__":
    root_dir = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data"
    images_dir = os.path.join(root_dir, "image")
    labels_dir = os.path.join(root_dir, "label")
    
    try:
        
        
        print("BONE-FOCUSED Multi-class Segmentation Training")
        print("="*80)
        print("This script focuses exclusively on optimizing bone class segmentation")
        print("="*80)
        
        # Run bone-focused training
        results = train_model(
            images_dir=images_dir,
            labels_dir=labels_dir,
            max_epochs=40,
            learning_rate=0.0001  # Lower learning rate for stability
        )
        
        if results:
            print("\n" + "="*80)
            print("BONE-FOCUSED TRAINING SUCCESSFULLY COMPLETED!")
            print("="*80)
            print(f"Best Bone Combined Dice: {results['best_bone_metric']}")
            print(f"Best Epoch: {results['best_epoch']}")
            print(f"Model Path: {results['model_path']}")
            print("\nFinal Class Dice Scores:")
            for class_name, score in results['final_class_dice_scores'].items():
                print(f"{class_name}: {score}")
        else:
            print("Training failed!")


        print(f"Training completed successfully!")
    except Exception as e:
        print(f"Training failed with error: {e}")