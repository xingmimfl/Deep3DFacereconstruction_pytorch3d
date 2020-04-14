import os
import sys
import glob
import numpy as np
from load_data import *
from scipy.io import loadmat, savemat
import cv2
import torch
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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
plt.switch_backend('agg')

device = torch.device('cuda:0')
torch.cuda.set_device(device)
facemodel = BFM()
facemodel.set_device(device)

def Split_coeff(coeff):
    id_coeff = coeff[:,:80] # identity(shape) coeff of dim 80
    ex_coeff = coeff[:,80:144] # expression coeff of dim 64
    tex_coeff = coeff[:,144:224] # texture(albedo) coeff of dim 80
    angles = coeff[:,224:227] # ruler angles(x,y,z) for rotation of dim 3
    gamma = coeff[:,227:254] # lighting coeff for 3 channel SH function of dim 27
    translation = coeff[:,254:] # translation coeff of dim 3
    return id_coeff,ex_coeff,tex_coeff,angles,gamma,translation

# compute face shape with identity and expression coeff, based on BFM model
# input: id_coeff with shape [1,80]
#        ex_coeff with shape [1,64]
# output: face_shape with shape [1,N,3], N is number of vertices
def Shape_formation(id_coeff, ex_coeff, facemodel, batch_size):
    face_shape = torch.einsum('ij,aj->ai', facemodel.idBase, id_coeff) + \
        torch.einsum('ij,aj->ai', facemodel.exBase, ex_coeff) + facemodel.meanshape
    face_shape = face_shape.view(batch_size,-1,3)
    # re-center face shape
    face_shape = face_shape - torch.mean(facemodel.meanshape.view(1,-1,3), dim=1, keepdims = True)
    return face_shape

# compute vertex texture(albedo) with tex_coeff
# input: tex_coeff with shape [1,N,3]
# output: face_texture with shape [1,N,3], RGB order, range from 0-255
def Texture_formation(tex_coeff,facemodel, batch_size):
    face_texture = torch.einsum('ij,aj->ai', facemodel.texBase, tex_coeff) + facemodel.meantex
    face_texture = face_texture.view(batch_size, -1, 3)

    return face_texture

# compute vertex normal using one-ring neighborhood
# input: face_shape with shape [1,N,3]
# output: v_norm with shape [1,N,3]
def Compute_norm(face_shape,facemodel, batch_size):
    face_id = facemodel.tri # vertex index for each triangle face, with shape [F,3], F is number of faces
    point_id = facemodel.point_buf # adjacent face index for each vertex, with shape [N,8], N is number of vertex
    device = face_id.get_device()
    #face_id = (face_id - 1).long()
    #point_id = (point_id - 1).long()
    face_id = face_id - 1
    point_id = point_id - 1

    v1 = face_shape[:,face_id[:,0],:]
    v2 = face_shape[:,face_id[:,1],:]
    v3 = face_shape[:,face_id[:,2],:]

    e1 = v1 - v2
    e2 = v2 - v3

    face_norm = torch.cross(e1,e2) # compute normal for each face
    #face_norm = torch.cat([face_norm.float(), torch.zeros([1,1,3])], dim=1) # concat face_normal with a zero vector at the end
    face_norm = torch.cat([face_norm, torch.zeros([batch_size,1,3]).to(device)], dim=1) # concat face_normal with a zero vector at the end

    v_norm = torch.sum(face_norm[:,point_id,:], dim=2) # compute vertex normal using one-ring neighborhood
    v_norm = v_norm/torch.norm(v_norm, p=2, dim=2, keepdim=True)
    return v_norm


def Compute_rotation_matrix(angles):
    device = angles.get_device()
    batch_size = angles.size()[0]
    #angle_x = angles[:,0][0]
    #angle_y = angles[:,1][0]
    #angle_z = angles[:,2][0]

    angle_x = angles[:,0]
    angle_y = angles[:,1]
    angle_z = angles[:,2]

    cosx = torch.cos(angle_x); sinx = torch.sin(angle_x)
    cosy = torch.cos(angle_y); siny = torch.sin(angle_y)
    cosz = torch.cos(angle_z); sinz = torch.sin(angle_z)

    rotation_x = torch.eye(3).repeat(2,1).view(batch_size, 3, 3).to(device)
    rotation_x[:, 1, 1] = cosx; rotation_x[:, 1, 2] = - sinx
    rotation_x[:, 2, 1] = sinx; rotation_x[:, 2, 2] = cosx

    rotation_y = torch.eye(3).repeat(2,1).view(batch_size, 3, 3).to(device)
    rotation_y[:, 0, 0] = cosy; rotation_y[:, 0, 2] = siny
    rotation_y[:, 2, 0] = -siny; rotation_y[:, 2, 2] = cosy

    rotation_z = torch.eye(3).repeat(2,1).view(batch_size, 3, 3).to(device)
    rotation_z[:,0,0] = cosz; rotation_z[:, 0, 1] = - sinz
    rotation_z[:,1,0] = sinz; rotation_z[:, 1, 1] = cosz

    # compute rotation matrix for X,Y,Z axis respectively
    rotation = torch.matmul(torch.matmul(rotation_z,rotation_y),rotation_x)
    rotation = rotation.permute(0,2,1)#transpose row and column (dimension 1 and 2)

    return rotation     



# compute vertex color using face_texture and SH function lighting approximation
# input: face_texture with shape [1,N,3]
#        norm with shape [1,N,3]
#        gamma with shape [1,27]
# output: face_color with shape [1,N,3], RGB order, range from 0-255
#         lighting with shape [1,N,3], color under uniform texture
def Illumination_layer(face_texture,norm,gamma):
    device = gamma.get_device()
    batch_size = gamma.size()[0]
    num_vertex = face_texture.size()[1]
    init_lit = torch.tensor([0.8,0,0,0,0,0,0,0,0]).to(device)
    gamma = gamma.view(-1,3,9)
    gamma = gamma + init_lit.view(1,1,9)

    # parameter of 9 SH function
    pi = 3.1415927410125732
    a0 = pi 
    a1 = 2 * pi/torch.sqrt(torch.tensor(3.0)).to(device)
    a2 = 2 * pi/torch.sqrt(torch.tensor(8.0)).to(device)
    c0 = 1 / torch.sqrt(torch.tensor(4 * pi)).to(device)
    c1 = torch.sqrt(torch.tensor(3.0)).to(device) / torch.sqrt(torch.tensor(4*pi))
    c2 = 3 * torch.sqrt(torch.tensor(5.0)).to(device) / torch.sqrt(torch.tensor(12*pi))

    Y0 = (a0*c0).view(1,1,1).repeat(batch_size, num_vertex,1) 
    Y1 = (-a1 * c1 * norm[:,:,1]).view(batch_size, num_vertex,1) 
    Y2 = (a1 * c1 * norm[:,:,2]).view(batch_size,num_vertex,1)
    Y3 = (-a1 * c1 * norm[:,:,0]).view(batch_size,num_vertex,1)
    Y4 = (a2 * c2 * norm[:,:,0] * norm[:,:,1]).view(batch_size,num_vertex,1)
    Y5 = (-a2 * c2 * norm[:,:,1] * norm[:,:,2]).view(batch_size,num_vertex,1)
    Y6 = (a2 * c2 * 0.5 / torch.sqrt(torch.tensor(3.0)) * (3* norm[:,:,2] ** 2 - 1)).view(batch_size,num_vertex,1)
    Y7 = (-a2 * c2 * norm[:,:,0] * norm[:,:,2]).view(batch_size,num_vertex,1)
    Y8 = (a2  * c2 * 0.5 * (norm[:,:,0] ** 2 - norm[:,:,1] ** 2)).view(batch_size,num_vertex,1)

    Y = torch.cat([Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8],dim=2)

    # Y shape:[batch,N,9].
    lit_r = torch.squeeze(torch.matmul(Y, torch.unsqueeze(gamma[:,0,:],2)),2) #[batch,N,9] * [batch,9,1] = [batch,N]
    lit_g = torch.squeeze(torch.matmul(Y, torch.unsqueeze(gamma[:,1,:],2)),2)
    lit_b = torch.squeeze(torch.matmul(Y, torch.unsqueeze(gamma[:,2,:],2)),2)

    # shape:[batch,N,3]
    face_color = torch.stack([lit_r*face_texture[:,:,0],lit_g*face_texture[:,:,1],lit_b*face_texture[:,:,2]],axis = 2)
    lighting = torch.stack([lit_r,lit_g,lit_b],axis = 2)*128
    return face_color, lighting

if __name__=="__main__":
    images_dir = "output"
    target_images_dir = "plot_images"
    if not os.path.exists(target_images_dir):
        os.makedirs(target_images_dir)

    batch_size = 2
    image_size = 512
    images_vec = []
    a_mat_path = os.path.join(images_dir, 'vd005.mat')
    a_mat = loadmat(a_mat_path)
    a_image = a_mat['cropped_img']
    a_image = cv2.resize(a_image, (image_size, image_size))
    images_vec.append(a_image)
    #print("a_image.size:\t", a_image.shape)
    #cv2.imwrite('a.jpg', a_image)
    a_coeff = a_mat['coeff']
    a_coeff = torch.tensor(a_coeff).to(device)
    
    b_mat_path = os.path.join(images_dir, 'vd006.mat')
    b_mat = loadmat(b_mat_path)
    b_image = b_mat['cropped_img']
    b_image = cv2.resize(b_image, (image_size, image_size))
    images_vec.append(b_image)
    b_coeff = b_mat['coeff']
    b_coeff = torch.tensor(b_coeff).to(device)
    coeff = torch.cat([a_coeff, b_coeff], dim=0)
    print('coeff.size:\t', coeff.size())

    id_coeff,ex_coeff,tex_coeff,angles,gamma,translation = Split_coeff(coeff)        

    face_shape = Shape_formation(id_coeff, ex_coeff, facemodel, batch_size)
    face_norm = Compute_norm(face_shape, facemodel, batch_size)

    rotation = Compute_rotation_matrix(angles)

    face_shape = torch.matmul(face_shape, rotation) ###旋转vertex
    face_shape = face_shape + translation.view(-1, 1, 3).repeat(1, face_shape.size()[1], 1)
    norm_r = torch.matmul(face_norm, rotation)
    face_texture = Texture_formation(tex_coeff, facemodel, batch_size)
    face_color, _ = Illumination_layer(face_texture, norm_r, gamma)


    #face_color = Textures(verts_rgb=face_texture.to(device)) #---改成pytorch3d中的Textures数据格式
    face_color = Textures(verts_rgb=face_color.to(device)) #---改成pytorch3d中的Textures数据格式
    face_index = facemodel.tri #---(num_of_faces, 3) 每个face对应的3个vertex序号
    face_index = face_index - 1 #---facemodel.tri 序号从1开始,改成0
    face_index = torch.stack([face_index, face_index], dim=0)
    #----为了适应Meshes的初始化,改成(1, face_num, 3)的形式
    #----long是mesh的输入要求
    #face_shape = face_shape.contiguous()
    #face_index = face_index.contiguous()
    print("face_shape.size:\t", face_shape.size())
    print('face_index.size:\t', face_index.size())
    mesh = Meshes(face_shape.to(device), face_index.to(device), face_color)

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
    )

    lights = PointLights(
        device=device, 
        ambient_color=((0.8, 0.8, 0.8),),
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
    images = renderer(mesh)
    print(images.size())

    transformed_face_shape = cameras.transform_points(face_shape)
    landmarks = transformed_face_shape[:, facemodel.keypoints, :]
    landmarks = ((landmarks + 1) * image_size - 1)/2.
    landmarks[:, :, :2] = image_size - landmarks[:, :, :2] #---x坐标和y坐标都需要倒置一下
    #print(landmarks)
    landmarks = landmarks.cpu().numpy()
    for i in range(batch_size):
        cropped_image = images_vec[i]
        cropped_image = torch.tensor(cropped_image).float().to(device)
        a_image = images[i, ..., :3]
        a_image = a_image + torch.min(a_image)
        a_image = a_image.clamp(0, 255)
        print("cropped_image.size:\t", cropped_image.size())
        print('a_image.size:\t', a_image.size())
        index = (a_image > 0)
        cropped_image[index] = a_image[index]
        a_image = cropped_image.clone()
        a_image = a_image.cpu().numpy()
        a_image = a_image[:, :, ::-1] #--rgb to bgr
        a_image_name = str(i) + ".png"
        cv2.imwrite(a_image_name, a_image)
        a_image = cv2.imread(a_image_name)
        for k in range(68):
            x,y,z = landmarks[i, k]
            x,y,z = int(x),int(y),int(z)
            cv2.circle(a_image, (x,y), 1, (255, 255,255), -1)
        cv2.imwrite(a_image_name, a_image)
