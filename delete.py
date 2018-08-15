from utils.load_imglist_Liu import ImageList

root = '/data/leolau/CelebA_Face/img_align_celeba_png/'
filelist = '/data/leolau/CelebA_Face/Anno/identity_CelebA.txt'
attrlist = ['/data/leolau/CelebA_Face/Anno/list_attr_celeba.txt',
            '/data/leolau/CelebA_Face/Anno/list_landmarks_align_celeba.txt']

imglist = ImageList(root, filelist, attrlist)
a, b, c = imglist.__test__(1)
print(a, b, c)
