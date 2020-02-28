import torch
import torch.nn as nn
import torch.nn
class Unet(nn.Module):
    '''U-Net Architecture'''
    def __init__(self,inp,out):
        super(Unet,self).__init__()

        self.c1=self.contracting_block(inp,8)
        self.c2=self.contracting_block(8,16)
        self.c3=self.contracting_block(16,32)
        self.c4=self.contracting_block(32,64)
        self.c5=self.contracting_block(64,128)

        self.maxpool=nn.MaxPool2d(2)
        self.upsample=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
        self.c6=self.contracting_block(64+128,64)
        self.c7=self.contracting_block(64+32,32)
        self.c8=self.contracting_block(32+16,16)
        self.c9=self.contracting_block(16+8,8)
        self.c10=nn.Conv2d(8,out,1)
        

    def contracting_block(self,inp,out,k=3):
        block =nn.Sequential(
            nn.Conv2d(inp,out,k,padding=1),
            nn.ReLU(inplace=True),
            torch.nn.Dropout2d(p=0.5, inplace=True)
            nn.Conv2d(out,out,k,padding=1),
            nn.ReLU(inplace=True),
            torch.nn.Dropout2d(p=0.5, inplace=True)
        )
        return block


    def forward(self,x):
        conv1=self.c1(x)#8,256,256
        conv2=self.c2(self.maxpool(conv1))#16,128,128
        conv3=self.c3(self.maxpool(conv2))#32,64,64
        conv4=self.c4(self.maxpool(conv3))#64,32,32
        
        conv5=self.c5(self.maxpool(conv4))#128,16,16
        #print(conv5.shape)
        #add dropout
        up1=self.upsample(conv5)#128,32,32
        one=self.c6(torch.cat([conv4,up1],axis=1))
        up2=self.upsample(one)
        two=self.c7(torch.cat([conv3,up2],axis=1))
        up3=self.upsample(two)
        three=self.c8(torch.cat([conv2,up3],axis=1))
        up4=self.upsample(three)
        four=self.c9(torch.cat([conv1,up4],axis=1))
        final=self.c10(four)
        return final

        


if __name__=="__main__":
    x=torch.ones(1,3,256,256)
    net=Unet(3,1)
    print(net(x).shape)
