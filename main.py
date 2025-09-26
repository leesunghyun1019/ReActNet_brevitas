import torch
from Network import Reactnet
from data import get_cifar10_loaders
from train import train


def main():
    # 하이퍼파라미터 설정
    BATCH_SIZE = 128
    EPOCHS = 50
    LEARNING_RATE = 0.01 
    NUM_CLASSES = 10

    # 데이터 로더 준비
    trainloader, testloader = get_cifar10_loaders(
        batch_size=BATCH_SIZE, 
        num_workers=4
    )

    # 모델 초기화
    model = Reactnet(
        num_classes=NUM_CLASSES, 
        imagenet=False,  # CIFAR-10이므로 False
        input_sign=True
    )

    # 모델 학습
    model = train(
        model, 
        trainloader, 
        testloader, 
        epochs=EPOCHS,
        lr=LEARNING_RATE
    )


if __name__ == "__main__":
    main()