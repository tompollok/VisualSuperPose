import torchreid
from LandmarksDataset import LandmarksDataset
import os
import argparse
import torch


# expected directory structure for root_data_dir:
# the three directories 0/0/0 are equal to the first 3 digits/letters in the name of the image
# --Google Landmarks Dataset root
# ----train
# ------0
# --------0
# ----------0
# ------------000abcdefghijklm.jpg


def train_model(data_dir="", sources="landmarks", targets="landmarks", height=224, width=224, batch_size_train=64,
                batch_size_test=100, model_name="resnet50", loss="softmax", pretrained=True, results_path="", epochs=10,
                learning_rate=0.0003, gpu=0, resume_model=""):

    datamanager = torchreid.data.ImageDataManager(
        root=data_dir,
        sources=sources,
        targets=targets,
        height=height,
        width=width,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        transforms=['random_flip', 'random_crop']
    )

    model = torchreid.models.build_model(
        name=model_name,
        num_classes=datamanager.num_train_pids,
        loss=loss,
        pretrained=True
    )

    model = model.to(f'cuda:{gpu}')

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=learning_rate
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=20
    )

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    if os.path.isfile(resume_model):
        start_epoch = torchreid.utils.resume_from_checkpoint(resume_model, model, optimizer)
    else:
        start_epoch=0

    engine.run(
        save_dir='log/' + model_name,
        max_epoch=epochs,
        start_epoch=start_epoch,
        eval_freq=10,
        print_freq=10,
        test_only=False,
    )
    return model


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="resnet50", help="Model architecture name")
    parser.add_argument("--data_dir", type=str, help="Path to train images directory")
    parser.add_argument("--results_path", type=str, help="Path to save result models to")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train for")
    parser.add_argument("--loss", type=str, choices=["softmax", "triplet"], help="Loss function to use for training")
    parser.add_argument("--combineall", type=bool, help="Whether to combine train, query, gallery", default=True)
    parser.add_argument("--dataset", type=str, default="landmarks")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume_training", type=str, default="")
    args = parser.parse_args()
    return args


def main():
    # example usage of this script: 'python train_test.py --model_name resnet50 --data_dir
    # /media/johanna/Volume/Studium/Semester_4/Bildauswertung_und_-fusion/GoogleLandmarksDataset --epochs 1 --loss
    # softmax --combineall True --batch_size 64'

    config = parse_arguments()
    torchreid.data.register_image_dataset(config.dataset, LandmarksDataset)
    model_name = config.model_name

    if not os.path.isdir(str(config.results_path)):
        config.results_path = 'log/' + model_name

    if not os.path.isdir(config.data_dir):
        raise IOError("Invalid data directory provided")

    print(config)

    model = train_model(data_dir=config.data_dir,
                        targets=config.dataset,
                        sources=config.dataset,
                        model_name=config.model_name,
                        height=224,
                        width=224,
                        batch_size_train=config.batch_size,
                        batch_size_test=config.batch_size,
                        loss=config.loss,
                        pretrained=config.pretrained,
                        results_path=config.results_path,
                        epochs=config.epochs,
                        learning_rate=config.learning_rate,
                        gpu=config.gpu,
                        resume_model=config.resume_training
                        )


if __name__ == "__main__":
    main()
