import matplotlib.pyplot as plt
import numpy as np
import torch
import albumentations as albu

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def Test_Pic(dataset,DEVICE, best_model):
    "Evaluation random images for example and visualizing them"

    for i in range(2):
        n = np.random.choice(len(dataset))

        image_vis = dataset[n][0].squeeze().transpose(1,2,0)
        image, gt_mask, impath = dataset[n]

        gt_mask=gt_mask.transpose(1,2,0)

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = pr_mask.cpu().squeeze(dim=1).numpy().round().transpose(1,2,0) #started with (1,1,256,256) tensor, ends with (256,256,1) numpy

        print(impath)
        visualize(
            image=image.squeeze().transpose(1,2,0),
            ground_truth_mask=gt_mask.squeeze(),
            predicted_mask=pr_mask.squeeze()
        )

def Test_Specipic(dataset,DEVICE, best_model,N=0):
    "Evaluation random images for example and visualizing them"
    "N = the number of the example in the test dataset (0-218)"
    for i in range(1):
        n = N
        image_vis = dataset[n][0].squeeze().transpose(1,2,0)
        image, gt_mask, impath = dataset[n]
        gt_mask=gt_mask.transpose(1,2,0)

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = pr_mask.cpu().squeeze(dim=1).numpy().round().transpose(1,2,0) # started with (1,1,256,256) tensor, ends with (256,256,1) numpy

        print(impath)
        visualize(
            image=image.squeeze().transpose(1,2,0),
            ground_truth_mask=gt_mask.squeeze(),
            predicted_mask=pr_mask.squeeze()

        )


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


