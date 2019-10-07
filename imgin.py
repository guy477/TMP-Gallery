import os
import numpy as np
import imageio
import matplotlib.pyplot as plt

class ImageHelper(object):
    def save_image(self, generated, epoch, directory):
        plt.figure(figsize=(8.5, 11))
        fig, axs = plt.subplots(1,2)
        count = 0
        """
        for i in range(2):
            for j in range(2):
                axs[i,j].imshow(generated[count, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                count += 1
        """
        for i in range(2):
                axs[i].imshow(generated[count, :,:,0], cmap='gray')
                axs[i].axis('off')
                count += 1
                
        fig.savefig("{}/{}.png".format(directory, epoch), dpi=350)
        plt.close()
        
    def makegif(self, directory):
        filenames = np.sort(os.listdir(directory))
        filenames = [ fnm for fnm in filenames if ".png" in fnm]
    
        with imageio.get_writer(directory + '/image.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(directory + filename)
                writer.append_data(image)