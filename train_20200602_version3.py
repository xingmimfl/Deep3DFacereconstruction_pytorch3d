import torch
import pdb
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import dataset
from config import *
import matplotlib.pyplot as plt
import numpy as np
import model
import os
import sys
import time
import cv2
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms

#----pytorch3d-------
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.transforms import Rotate, Translate, RotateAxisAngle
from pytorch3d.structures import Meshes, Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturedSoftPhongShader,
    HardPhongShader,
    SoftPhongShader,
)
#-----pytorch3d--------

#---deep3d----

#device = torch.device('cuda:%d' % DEVICE_IDS[0])
#device = torch.cuda.set_device(2)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda')

batch_size = BATCH_SIZE
image_size = CROP_SIZE

#--------face id--------


#-----render-----
R, T = look_at_view_transform(eye=((0,0,10.0),), at=((0,0,0),), up=((0,1.,0),))
cameras = OpenGLPerspectiveCameras(
    device=device,
    znear = 0.01,
    zfar = 50.,
    aspect_ratio = 1.,
    fov = 12.5936,
    R=R, T=T
)

raster_settings = RasterizationSettings(
    image_size=image_size,
    blur_radius=0.0,
    faces_per_pixel=1,
    bin_size=None,
)

lights = PointLights(
    device=device,
    ambient_color=((0.5, 0.5, 0.5),),
    location=((0.0, 0.0, 1e5),)
)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(
        device=device,
        cameras=cameras,
        lights=lights
    )
)
#-----render-----

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

mean=[0.485, 0.456, 0.406]
mean = torch.tensor(mean, dtype=torch.float32)
std=[0.229, 0.224, 0.225]
std = torch.tensor(std, dtype=torch.float32)

def main():
    setup_information()
    landmarks_model = _model_init()
    train_loader = get_dataset()


    for p in landmarks_model.resnet.conv1.parameters():p.requires_grad=False
    for p in landmarks_model.resnet.bn1.parameters(): p.requires_grad=False
    for p in landmarks_model.resnet.layer1.parameters(): p.requires_grad=False
    #for p in landmarks_model.resnet.layer1.0.parameters(): p.requires_grad=False
    params = []
    #params = list(net.parameters())
    for p in list(landmarks_model.parameters()):
        if p.requires_grad == False: continue
        params.append(p)
    optimizer = optim.RMSprop(params, lr=LEARNING_RATE)
    scheduler = MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.2)
    landmark_avg_loss = AverageMeter('landmark_avg_loss', ':.4e')
    ks_landmark_avg_loss = AverageMeter('ks_landmark_avg_loss', ':.4e')
    coeff_avg_loss = AverageMeter('coeff_avg_loss', ':.4e')
    pixel_avg_loss = AverageMeter('pixel_avg_loss', ':.4e')
    skin_mask_avg_loss = AverageMeter('skin_mask_loss', ':.4e')
    avg_loss = AverageMeter('loss', ':.4e')
    

    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    #trans = transforms.Compose([
    #    transforms.ToTensor(),
    #    normalize,
    #])

    for epoch in range(MAX_EPOCH):
        epoch_time = time.time()
        scheduler.step()
        count = 0
        for i_batch, _sample_batched in enumerate(train_loader):
            _inputs = _sample_batched[0] #--- batch_size x 3 x 224 x 224, bgr, 0, 255
            _bgr_images = _sample_batched[1].float() #--batch_size x 224 x 224 x 3, bgr, [0, 255]
            _mask_images = _sample_batched[2].ge(200)
            _gt_pts = _sample_batched[3]

            _inputs = _inputs/255.
            _inputs = _inputs.sub_(mean[:, None, None]).div_(std[:, None, None])
            _gt_pts = _gt_pts / CROP_SIZE * image_size
            _gt_pts = _gt_pts.view(batch_size, -1, 3)

            _inputs = _inputs.to(device)
            _bgr_images = _bgr_images.to(device)
            _mask_images = _mask_images.to(device)
            _gt_pts = _gt_pts.to(device)
            _render_images = _bgr_images.clone()

            coeff = landmarks_model(_inputs)
            #-----rendering-----------------------
            id_coeff,ex_coeff,tex_coeff,angles,gamma,translation = landmarks_model.Split_coeff(coeff)
            coeff_loss = landmarks_model.get_coeff_loss(id_coeff, ex_coeff, tex_coeff)

            face_shape = landmarks_model.Shape_formation(id_coeff, ex_coeff, batch_size)
            face_norm = landmarks_model.Compute_norm(face_shape, batch_size)
            rotation = landmarks_model.Compute_rotation_matrix(angles)
            face_shape = torch.matmul(face_shape, rotation) ###旋转vertex
            face_shape = face_shape + translation.view(-1, 1, 3).repeat(1, face_shape.size()[1], 1)
            norm_r = torch.matmul(face_norm, rotation)
            face_texture = landmarks_model.Texture_formation(tex_coeff, batch_size)
            face_color, _ = landmarks_model.Illumination_layer(face_texture, norm_r, gamma)

            face_color = Textures(verts_rgb=face_color.to(device)) #---改成pytorch3d中的Textures数据格式
            skin_mask_color = face_texture[:, landmarks_model.skinmask,:]
            skin_mask_loss = landmarks_model.get_skin_mask_loss(skin_mask_color)
            
            mesh = Meshes(face_shape.to(device), landmarks_model.face_index, face_color)            
            
            #---landmarks------
            transformed_face_shape = cameras.transform_points(face_shape)
            landmarks = transformed_face_shape[:, landmarks_model.facemodel.keypoints, :]
            landmarks = ((landmarks + 1) * image_size - 1)/2.
            landmarks[:, :, :2] = image_size - landmarks[:, :, :2] #---x坐标和y坐标都需要倒置一下
            landmark_loss = landmarks_model.get_landmark_loss(_gt_pts[:,:,:2], landmarks[:, :, :2])

            #-------rendered images---
            images = renderer(mesh)
            images = images[:, :, :, :3] #---get images
            images = images[:, :, :, [2,1,0]] #---rgb to bgr
            images_clone = images.clone()
            #images = images.clamp(0, 255)
            index = (images > 0)
            _render_images[index] = images[index]
            target_images_dir = "debug_images_dir"
            if not os.path.exists(target_images_dir):
                os.makedirs(target_images_dir)

            image_leve_loss = landmarks_model.get_image_level_loss(_render_images, _bgr_images, _mask_images)
            #landmark_loss = 1.6e-3 * landmark_loss
            #image_leve_loss = 1.9 * image_leve_loss
            #coeff_loss = 3e-4 * coeff_loss
            #skin_mask_loss = 5 * skin_mask_loss
            landmark_loss = 0.5 * landmark_loss
            image_leve_loss =  0.1 * image_leve_loss
            coeff_loss = coeff_loss
            skin_mask_loss = skin_mask_loss
            loss = image_leve_loss +  coeff_loss + skin_mask_loss + landmark_loss
            #_bgr_images = _bgr_images.cpu().detach().numpy()
            #for i in range(batch_size):
            #   a_image = _bgr_images[i]
            #   a_target_image_path = os.path.join(target_images_dir, str(i) + '.jpg')
            #   cv2.imwrite(a_target_image_path, a_image)
            #sys.exit(0)

            #_render_images = _render_images.cpu().detach().numpy()
            #for i in range(batch_size):
            #   a_image = _render_images[i]
            #   a_target_image_path = os.path.join(target_images_dir, str(i) + '.jpg')
            #   cv2.imwrite(a_target_image_path, a_image)            

            avg_loss.update(loss.detach().item())
            landmark_avg_loss.update(landmark_loss.detach().item())
            skin_mask_avg_loss.update(skin_mask_loss.detach().item())
            coeff_avg_loss.update(coeff_loss.detach().item())
            pixel_avg_loss.update(image_leve_loss.detach().item())
            if count % 100 == 0:
                print('Iter: [%d, %5d]' % (epoch, i_batch))
                print(' Iter: [%d, %5d]' % (epoch, i_batch) + ' landmark_loss' + ': %.3e' % landmark_avg_loss.avg)
                print(' Iter: [%d, %5d]' % (epoch, i_batch) + ' coeff_loss' + ': %.3e' % coeff_avg_loss.avg)
                print(' Iter: [%d, %5d]' % (epoch, i_batch) + ' skin_mask_loss' + ': %.3e' % skin_mask_avg_loss.avg)
                print(' Iter: [%d, %5d]' % (epoch, i_batch) + ' image_leve_loss' + ': %.3e' % pixel_avg_loss.avg)
                print(' Iter: [%d, %5d]' % (epoch, i_batch) + ' loss' + ': %.3e' % avg_loss.avg)
                print('\n')
                _render_images = _render_images.cpu().detach().numpy()
                _bgr_images = _bgr_images.cpu().detach().numpy()
                for i in range(batch_size):
                    a_image = _render_images[i]
                    b_image = _bgr_images[i]
                    c_image = np.concatenate((a_image, b_image), axis=1)
                    a_target_image_path = os.path.join(target_images_dir, str(i) + '.jpg')
                    cv2.imwrite(a_target_image_path, c_image)                

                landmark_avg_loss = AverageMeter('landmark_avg_loss', ':.4e')
                skin_mask_loss = AverageMeter('skin_mask_loss', ':.4e')
                pixel_avg_loss = AverageMeter('pixel_avg_loss', ':.4e')
                avg_loss = AverageMeter('loss', ':.4e')

            if count % 500 == 0:
                a_save_name = "_".join([SUFFIX, 'iter', 'epoch', '%d' % epoch, 'i_batch', '%d' % i_batch]) + '.pth'
                a_save_path = os.path.join(WRITE_SNAPSHOT_PATH, a_save_name)
                torch.save(landmarks_model.state_dict(), a_save_path)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1


def setup_information():
    print('LEARNING_RATE:', LEARNING_RATE)
    print('MILESTONES:', MILESTONES)
    print('SUFFIX:', SUFFIX)
    print('BATCH_SIZE:', BATCH_SIZE)
    print('DEVICE_IDS:', DEVICE_IDS)
    print('FINETUNE:', FINETUNE)
    print('NUM_WORKERS:', NUM_WORKERS)


def get_dataset():
    pass

def _model_init():
    landmarks_model = model.LandmarksModel()
    landmarks_model.to(device)
    landmarks_model.set_device(device)
    return landmarks_model


if __name__ == '__main__':
    main()
