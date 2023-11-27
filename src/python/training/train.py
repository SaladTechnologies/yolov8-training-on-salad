from ultralytics import YOLO
import torch
import os
import logging.config
import argparse


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--output_model_name",
            default=os.environ.get("OUTPUT_MODEL_NAME", "yolov8_logos"),
            type=str,
        ),
        parser.add_argument(
            "--base_model",
            default=os.environ.get("BASE_MODEL", "yolov8n.pt"),
        ),
        parser.add_argument(
            "--epochs",
            default=os.environ.get("EPOCHS", 50),
            type=int,
        ),
        parser.add_argument(
            "--batch",
            default=os.environ.get("BATCH", 8),
            type=int,
        ),
        parser.add_argument(
            "--image_size",
            default=os.environ.get("IMAGE_SIZE", 640),
            type=int,
        ),
        args = parser.parse_args()

        output_model_name = args.output_model_name
        last_model_path = f"runs/detect/{output_model_name}/weights/last.pt"
        base_model = args.base_model

        # Check for CUDA device and set it
        device = "0" if torch.cuda.is_available() else "cpu"
        if device == "0":
            torch.cuda.set_device(0)

        # check if we have weights for the model in runs/detect/output_model_name/weights
        # if not, download them
        if os.path.isfile(last_model_path):
            model = YOLO(last_model_path)
            model.train(resume=True)
        else:
            # Load the nano model.
            model = YOLO(base_model)

            # Train the model
            model.train(
                data="yolo8.yaml",
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.image_size,
                name=output_model_name,
                save=True,
                save_period=1,
            )

        # Evaluate the model's performance on the validation set
        model.val()

        # Create a done.txt file to indicate that the training is done
        with open("done.txt", "w") as f:
            f.write("done")

    except Exception as exc:
        logging.error(exc)
        raise
    finally:
        print("done")


if __name__ == "__main__":
    main()
