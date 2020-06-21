import sys
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from torch.nn.parameter import Parameter
from config import *
from load_data import *
from resnet import resnet50

class LandmarksModel(nn.Module):
    dump_patches = True

    def __init__(self):
        super(LandmarksModel, self).__init__()
        self.resnet = resnet50()
        pretrained_model = "resnet50-19c8e357.pth"
        pretrained_state_dict = torch.load(pretrained_model)
        state_dict = self.resnet.state_dict()
        for key, value in pretrained_state_dict.items():
            if "fc" in key: continue
            print(key)
            state_dict[key] = pretrained_state_dict[key]
        self.resnet.load_state_dict(state_dict)


        self.facemodel = BFM()

        #----face_index----
        face_index = self.facemodel.tri #---(num_of_faces, 3) 每个face对应的3个vertex序号
        face_index = face_index - 1 #---facemodel.tri 序号从1开始,改成0
        self.face_index = face_index.repeat(BATCH_SIZE, 1, 1)


        #---skinmask---
        self.skinmask = self.facemodel.skinmask

        ####----mean value---
        self.mean_id_coeff = torch.tensor(np.load('coeff_mean/id_coeff_avg.npy'))
        self.mean_ex_coeff = torch.tensor(np.load('coeff_mean/ex_coeff_avg.npy'))
        self.mean_tex_coeff = torch.tensor(np.load('coeff_mean/tex_coeff_avg.npy'))
        #self.mead_gamma_coeff = torch.tensor(np.load('coeff_mean/gamma_avg.npy'))

    def forward(self, x):
        output = self.resnet(x)
        return output



    def set_device(self, device):
        self.mean_id_coeff = self.mean_id_coeff.to(device)        
        self.mean_ex_coeff = self.mean_ex_coeff.to(device)
        self.mean_tex_coeff = self.mean_tex_coeff.to(device)
        self.face_index = self.face_index.to(device)
        self.facemodel.set_device(device)


    def Split_coeff(self, coeff):
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
    def Shape_formation(self, id_coeff, ex_coeff, batch_size):
        face_shape = torch.einsum('ij,aj->ai', self.facemodel.idBase, id_coeff) + \
            torch.einsum('ij,aj->ai', self.facemodel.exBase, ex_coeff) + self.facemodel.meanshape
        face_shape = face_shape.view(batch_size,-1,3)
        # re-center face shape
        face_shape = face_shape - torch.mean(self.facemodel.meanshape.view(1,-1,3), dim=1, keepdims = True)
        return face_shape

    # compute vertex texture(albedo) with tex_coeff
    # input: tex_coeff with shape [1,N,3]
    # output: face_texture with shape [1,N,3], RGB order, range from 0-255
    def Texture_formation(self, tex_coeff, batch_size):
        face_texture = torch.einsum('ij,aj->ai', self.facemodel.texBase, tex_coeff) + self.facemodel.meantex
        face_texture = face_texture.view(batch_size, -1, 3)
        return face_texture


    # compute vertex normal using one-ring neighborhood
    # input: face_shape with shape [1,N,3]
    # output: v_norm with shape [1,N,3]
    def Compute_norm(self, face_shape, batch_size):
        face_id = self.facemodel.tri # vertex index for each triangle face, with shape [F,3], F is number of faces
        point_id = self.facemodel.point_buf # adjacent face index for each vertex, with shape [N,8], N is number of vertex
        device = face_id.get_device()
        face_id = (face_id - 1).long()
        point_id = (point_id - 1).long()
        #face_id = face_id - 1
        #point_id = point_id - 1

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

    def Illumination_layer(self, face_texture,norm,gamma):
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

    def Compute_rotation_matrix(self, angles):
        device = angles.get_device()
        batch_size = angles.size()[0]

        angle_x = angles[:,0]
        angle_y = angles[:,1]
        angle_z = angles[:,2]

        cosx = torch.cos(angle_x); sinx = torch.sin(angle_x)
        cosy = torch.cos(angle_y); siny = torch.sin(angle_y)
        cosz = torch.cos(angle_z); sinz = torch.sin(angle_z)

        rotation_x = torch.eye(3).repeat(batch_size,1).view(batch_size, 3, 3).to(device)
        rotation_x[:, 1, 1] = cosx; rotation_x[:, 1, 2] = - sinx
        rotation_x[:, 2, 1] = sinx; rotation_x[:, 2, 2] = cosx

        rotation_y = torch.eye(3).repeat(batch_size,1).view(batch_size, 3, 3).to(device)
        rotation_y[:, 0, 0] = cosy; rotation_y[:, 0, 2] = siny
        rotation_y[:, 2, 0] = -siny; rotation_y[:, 2, 2] = cosy

        rotation_z = torch.eye(3).repeat(batch_size,1).view(batch_size, 3, 3).to(device)
        rotation_z[:,0,0] = cosz; rotation_z[:, 0, 1] = - sinz
        rotation_z[:,1,0] = sinz; rotation_z[:, 1, 1] = cosz

        # compute rotation matrix for X,Y,Z axis respectively
        rotation = torch.matmul(torch.matmul(rotation_z,rotation_y),rotation_x)
        rotation = rotation.permute(0,2,1)#transpose row and column (dimension 1 and 2)
        return rotation

    def get_landmark_loss(self, outputs, labels):
        mouth_index = range(48, 68)
        left_index = range(48)
        outputs_mouth = outputs[:, mouth_index]
        outputs_left = outputs[:, left_index]
        labels_mouth = labels[:, mouth_index]
        labels_left = labels[:, left_index]
        results = torch.mean((outputs_mouth - labels_mouth) ** 2) + 20 * torch.mean((outputs_left - labels_left) ** 2)
        results = results/ 68.
        return results

    def get_image_level_loss(self, render_images, gt_images, mask_images):
        render_images_tmp = render_images / 255.
        gt_images_tmp = gt_images / 255 
        render_images_tmp = torch.masked_select(render_images_tmp, mask_images)
        gt_images_tmp = torch.masked_select(gt_images_tmp, mask_images)
        delta = (render_images_tmp  - gt_images_tmp)
        loss = torch.norm(delta)
        #loss = torch.sqrt(delta**2)
        return loss
    
    def get_perceptual_loss(self, render_images, gt_images):
        pass

    def get_coeff_loss(self, id_coeff, ex_coeff, tex_coeff):
        coeff_loss = torch.mean(id_coeff**2) + 0.8*torch.mean(ex_coeff**2) + 1.7e-3*torch.mean(tex_coeff**2)

        return coeff_loss

    def get_skin_mask_loss(self, skin_mask_color):
        skin_mask_color = skin_mask_color/255.
        r = skin_mask_color[:, :, 0]
        g = skin_mask_color[:, :, 1]
        b = skin_mask_color[:, :, 2]
        loss = torch.mean(r.std() + g.std() + b.std())
        return loss

if __name__=="__main__":
    models = LandmarksModel()
    print(models)
    input = torch.rand((1, 3, 224, 224))
    output = models(input)
    print("output.size:\t", output.size())
    for p in models.resnet.parameters():
        print(p)
