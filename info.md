# History 平滑
    
    from fastai.vision.image import open_image
    img = open_image(file, 'RGB') 
       
    plt.hist(img.data.cpu().numpy().flatten(), bins=100)
    plt.show()
    
    
    img = img.apply_tfms(do_resolve=True, tfms=None, size=224)
    plt.hist(img.data.cpu().numpy().flatten(), bins=100)
    plt.show()