#!/usr/bin/env python
# coding: utf-8

# In[1]:


from basic_utils import *


# In[2]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'On {device}')
Tensor  = torch.cuda.FloatTensor

# data (img)
img_height = 256
img_width = 256
channels = 3

# training
epoch = 0 # epoch to start training from
n_epochs = 50 # number of epochs of training
batch_size = 1 # size of the batches
lr = 0.0002 # adam : learning rate
b1 = 0.5 # adam : decay of first order momentum of gradient
b2 = 0.999 # adam : decay of first order momentum of gradient
decay_epoch = 3 # suggested default : 100 (suggested 'n_epochs' is 200)
                 # epoch from which to start lr decay

    
# transforms_ = [
#     transforms.Resize(int(img_height*1.12), Image.BICUBIC),
#     transforms.RandomCrop((img_height, img_width)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ]


# In[17]:


class CyDataset(Dataset):
    def __init__(self):

        self.D = []
        self.L = []
                
        with open('/datacommons/carlsonlab/srs108/old/ol/Delhi_clean.pkl', "rb") as fp:
            for station in tqdm(pkl.load(fp)):
                self.D.append(tuple((station['Image'][:,:,:3], station['PM25'])))
                
        with open('/datacommons/carlsonlab/srs108/old/ol/Beijing_clean.pkl', "rb") as fp:
            for station in tqdm(pkl.load(fp)):
                self.L.append(tuple((station['Image'][:,:,:3], station['PM25'])))
#                 for datapoint in station:
#                     luck_img = datapoint['Image'][:,:,:3]
#                     if luck_img.shape == (224, 224,3):  
#                         self.L.append(tuple((luck_img, datapoint['PM'])))
                        
        self.D = random.choices(self.D, k= len(self.L))
        
    def __len__(self): return (len(self.D))
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

            #Delhi normalization
        transform  = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.Pad(16),
                            transforms.ToTensor(),
#                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        d_img = self.D[idx][0]
        d_img = transform(d_img)        
        l_img = self.L[idx][0]
        l_img = transform(l_img)
        
        sample = {
              'A': d_img,
#               'D pm' : torch.tensor(self.D[idx][1]),
              'B': l_img,
#               'L pm' : torch.tensor(self.L[idx][1])
        }
        return sample


# In[18]:


tr = CyDataset()


# In[22]:


train, val = train_test_split(tr,test_size=0.2, random_state=69)


# In[23]:


dataloader = DataLoader(
    train,
    batch_size=1, # 1
    shuffle=True)

val_dataloader = DataLoader(
    val,
    batch_size=1, # 1
    shuffle=True)


# In[24]:


# class ImageDataset(Dataset):
#     def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
#         self.transform = transforms.Compose(transforms_)
#         self.unaligned = unaligned
#         self.mode = mode
#         if self.mode == 'train':
#             self.files_A = sorted(glob.glob(os.path.join(root+'/thil')+'/*.*'))
#             self.files_B = sorted(glob.glob(os.path.join(root+'/americanclub')+'/*.*'))
#         elif self.mode == 'test':
#             self.files_A = sorted(glob.glob(os.path.join(root+'/thil')+'/*.*'))
#             self.files_B = sorted(glob.glob(os.path.join(root+'/americanclub')+'/*.*'))

#     def  __getitem__(self, index):
#         image_A = Image.open(self.files_A[index % len(self.files_A)])
        
#         if self.unaligned:
#             image_B = Image.open(self.files_B[np.random.randint(0, len(self.files_B)-1)])
#         else:
#             image_B = Image.open(self.files_B[index % len(self.files_B)])
#         if image_A.mode != 'RGB':
#             image_A = to_rgb(image_A)
#         if image_B.mode != 'RGB':
#             image_B = to_rgb(image_B)
            
#         item_A = self.transform(image_A)
#         item_B = self.transform(image_B)
#         return {'A':item_A, 'B':item_B}
    
#     def __len__(self):
#         return max(len(self.files_A), len(self.files_B))


# In[6]:


# root = '/datacommons/carlsonlab/srs108/planet_imgs/'

# dataloader = DataLoader(
#     ImageDataset(root, transforms_=transforms_, unaligned=True),
#     batch_size=1, # 1
#     shuffle=True,)

# val_dataloader = DataLoader(
#     ImageDataset(root, transforms_=transforms_, unaligned=True, mode='test'),
#     batch_size=1,
#     shuffle=True,
# )


# In[8]:


criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()


# In[9]:


input_shape = (channels, img_height, img_width) # (3,256,256)
n_residual_blocks = 9 # suggested default, number of residual blocks in generator

G_AB = GeneratorResNet(input_shape, n_residual_blocks)
G_BA = GeneratorResNet(input_shape, n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)


# In[10]:


cuda = torch.cuda.is_available()

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()


# In[11]:


G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)
print()


# In[12]:


optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1,b2)
)

optimizer_D_A = torch.optim.Adam(
    D_A.parameters(), lr=lr, betas=(b1,b2)
)
optimizer_D_B = torch.optim.Adam(
    D_B.parameters(), lr=lr, betas=(b1,b2)
)


# In[13]:



lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G,
    lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
)

lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A,
    lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B,
    lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
)


# In[ ]:


def sample_images(dataloader, epochs, iters, save = False):
    source = next(iter(dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = source['A'].type(Tensor) # A : monet
    fake_B = G_AB(real_A).detach()
    real_B = source['B'].type(Tensor) # B : photo
    fake_A = G_BA(real_B).detach()

    recon_A = G_BA(fake_B).detach()
    recon_B = G_AB(fake_A).detach()
    
    real_A = make_grid(real_A, nrow=5, normalize=True, scale_each=True, padding=1)
    fake_B = make_grid(fake_B, nrow=5, normalize=True, scale_each=True, padding=1)
    real_B = make_grid(real_B, nrow=5, normalize=True, scale_each=True, padding=1)
    fake_A = make_grid(fake_A, nrow=5, normalize=True, scale_each=True, padding=1)
    recon_A = make_grid(recon_A, nrow=5, normalize=True, scale_each=True, padding=1)
    recon_B = make_grid(recon_B, nrow=5, normalize=True, scale_each=True, padding=1)
    # Arange images along y-axis    
    image_grid = torch.cat((real_A, fake_B, recon_A, real_B, fake_A, recon_B), 2)
    plt.imshow(image_grid.cpu().permute(1,2,0))
    plt.title('Real Delhi vs Fake Delhi vs Recon Delhi | Real Beijing vs Fake Beijing vs Recon Beijing')
    plt.axis('off')
    plt.gcf().set_size_inches(12, 9)
    if save:
        plt.savefig(os.path.join('Figure_PDFs', f'epoch_{str(e+1)}_iter{str(i+1)}.png'), bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.show();


# In[35]:


history = {'epoch':[],'G_loss':[],'adv_loss': [],'cyc_loss': [], 'idt_loss': [], 'D_loss':[], 'batch':[]}

for epoch in range(epoch, n_epochs):
    for i, batch in enumerate(tqdm(dataloader)):
        
        # Set model input
        real_A = batch['A'].type(Tensor)
        real_B = batch['B'].type(Tensor)
        
        # Adversarial ground truths
        valid = Tensor(np.ones((real_A.size(0), *D_A.output_shape))) # requires_grad = False. Default.
        fake = Tensor(np.zeros((real_A.size(0), *D_A.output_shape))) # requires_grad = False. Default.
        
# -----------------
# Train Generators
# -----------------
        G_AB.train() # train mode
        G_BA.train() # train mode
        
        optimizer_G.zero_grad() # Integrated optimizer(G_AB, G_BA)
        
        # Identity Loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A) # If you put A into a generator that creates A with B,
        loss_id_B = criterion_identity(G_AB(real_B), real_B) # then of course A must come out as it is.
                                                             # Taking this into consideration, add an identity loss that simply compares 'A and A' (or 'B and B').
        loss_identity = (loss_id_A + loss_id_B)/2
        
        # GAN Loss
        fake_B = G_AB(real_A) # fake_B is fake-photo that generated by real monet-drawing
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid) # tricking the 'fake-B' into 'real-B'
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid) # tricking the 'fake-A' into 'real-A'
        
        loss_GAN = (loss_GAN_AB + loss_GAN_BA)/2
        
        # Cycle Loss
        recov_A = G_BA(fake_B) # recov_A is fake-monet-drawing that generated by fake-photo
        loss_cycle_A = criterion_cycle(recov_A, real_A) # Reduces the difference between the restored image and the real image
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        
        loss_cycle = (loss_cycle_A + loss_cycle_B)/2
        
# ------> Total Loss
        loss_G = loss_GAN + (10.0*loss_cycle) + (5.0*loss_identity) # multiply suggested weight(default cycle loss weight : 10, default identity loss weight : 5)
        
        loss_G.backward()
        optimizer_G.step()
        
# -----------------
# Train Discriminator A
# -----------------
        optimizer_D_A.zero_grad()
    
        loss_real = criterion_GAN(D_A(real_A), valid) # train to discriminate real images as real
        loss_fake = criterion_GAN(D_A(fake_A.detach()), fake) # train to discriminate fake images as fake
        
        loss_D_A = (loss_real + loss_fake)/2
        
        loss_D_A.backward()
        optimizer_D_A.step()

# -----------------
# Train Discriminator B
# -----------------
        optimizer_D_B.zero_grad()
    
        loss_real = criterion_GAN(D_B(real_B), valid) # train to discriminate real images as real
        loss_fake = criterion_GAN(D_B(fake_B.detach()), fake) # train to discriminate fake images as fake
        
        loss_D_B = (loss_real + loss_fake)/2
        
        loss_D_B.backward()
        optimizer_D_B.step()
        
# ------> Total Loss
        loss_D = (loss_D_A + loss_D_B)/2
    
# -----------------
# Show Progress
# -----------------
        if (i+1) % 500 == 0:
            sample_images(val_dataloader, n_epochs, i, save = False)
            print('[Epoch %d/%d] [Batch %d/%d] [D loss : %f] [G loss : %f - (adv : %f, cycle : %f, identity : %f)]'
                    %(epoch+1,n_epochs,       # [Epoch -]
                      i+1,len(dataloader),   # [Batch -]
                      loss_D.item(),       # [D loss -]
                      loss_G.item(),       # [G loss -]
                      loss_GAN.item(),     # [adv -]
                      loss_cycle.item(),   # [cycle -]
                      loss_identity.item(),# [identity -]
                     ))
            
            history['G_loss'].append(loss_G.item())
            history['D_loss'].append(loss_D.item())
            history['batch'].append(i+1)
            history['epoch'].append(epoch+1)
            history['adv_loss'].append(loss_GAN.item())
            history['cyc_loss'].append(loss_cycle.item())
            history['idt_loss'].append(loss_identity.item())
            


# In[ ]:


df = pd.DataFrame(history)
df.to_csv('history.csv', index=False)

