import torch 
import torch.nn as nn
from ET.convGRU import ConvGRU
import numpy as np

class ET_convGRU(nn.Module):
    def __init__(self, config = None):
        super(ET_convGRU, self).__init__()
        self.enhance = ConvGRU(input_size=(6, 6),
                        input_dim=512,
                        hidden_dim=512,
                        num_layers=2,
                        kernel_size=(1, 1),
                        bias=True,
                        batch_first=True,
                        dtype=torch.cuda.FloatTensor,
                        return_all_layers = False)


    def forward(self, x, x0):
        # t = torch.cat((x, y, z), 0)
        # x0 is residual
        response = self.enhance(x)[0][0][:,-1]
        response += x0
        return response


class ET256(nn.Module):
    def __init__(self, config=None):
        super(ET256, self).__init__()
        self.enhance = nn.Sequential(
            nn.Conv2d(768, 96, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, 1),
        )
    def forward(self, x, x0):
        #t = torch.cat((x, y, z), 0)
        # x0 is residual
        response = self.enhance(x)
        response += x0
        return response

if __name__ == '__main__':

    # network test
    net = ET256()
    net.eval()
    
class ET512_dw(nn.Module):
    def __init__(self, config=None):
        super(ET512_dw, self).__init__()
        self.enhance = nn.Sequential(
            nn.Conv2d(798, 798, 1,groups=798),
            # nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            nn.Conv2d(798, 256, 1),
        )
    def forward(self, x, x0):
        #t = torch.cat((x, y, z), 0)
        # x0 is residual
        response = self.enhance(x)
        response += x0
        return response
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ET512(nn.Module):
    def __init__(self, config=None):
        super(ET512, self).__init__()
        self.enhance = nn.Sequential(
            nn.Conv2d(1536, 1536, 1,groups=1536),
            nn.ReLU(inplace=True),
            nn.Conv2d(1536, 512, 1),
        )
    def forward(self, x, x0):
        #t = torch.cat((x, y, z), 0)
        # x0 is residual
        response = self.enhance(x)
        response += x0
        return response
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ET(nn.Module):
    def __init__(self, config=None):
        super(ET, self).__init__()
        self.enhance = nn.Sequential(
            nn.Conv2d(768, 96, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, 1),            
            #nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )

    def forward(self, x):
        #t = torch.cat((x, y, z), 0)
        response = self.enhance(x)
        return response

class ET_3(nn.Module):
    def __init__(self, config=None):
        super(ET_3, self).__init__()
        self.et_0 = nn.Sequential(
            nn.Conv2d(768, 96, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, 1),
            #nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )
        self.et_1 = nn.Sequential(
            nn.Conv2d(768, 96, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, 1),
            # nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )
        self.et_2 = nn.Sequential(
            nn.Conv2d(768, 96, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, 1),
            # nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )

    def forward(self, x_0,x_1,x_2,x0_0,x0_1,x0_2):
        # t = torch.cat((x, y, z), 0)
        # x0 is residual
        response_0 = self.et_0(x_0)
        response_0 += x0_0
        response_1 = self.et_1(x_1)
        response_1 += x0_1
        response_2= self.et_2(x_2)
        response_2 += x0_2

        return {
            'response_0':  response_0,
            'response_1':  response_1,
            'response_2':  response_2
        }

class ET_seq(nn.Module):
        def __init__(self, config=None):
            super(ET_seq, self).__init__()
            self.et_0 = nn.Sequential(
                nn.Conv2d(768, 512, 1),
                nn.ReLU(inplace=True),

                # # nn.Conv2d(192, 512, 1),
                # # nn.ReLU(inplace=True),
                 nn.Conv2d(512,256, 1),
                # nn.BatchNorm2d(128),
                # nn.ReLU(inplace=True),
                #
                # nn.Conv2d(128, 256, 1)
                # nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
            )

        def forward(self, x_0, x0_0):
            # t = torch.cat((x, y, z), 0)
            # x0 is residual
            response_0 = self.et_0(x_0)
            response_0 += x0_0

            return response_0

def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
#

#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
class ET_test(nn.Module):
    def __init__(self, config=None):
        super(ET_test, self).__init__()
        self.et_0 = nn.Sequential(
            nn.Conv2d(768, 96, 1),
            # nn.BatchNorm2d(2560),
            nn.ReLU(inplace=True),
            # nn.Conv2d(192, 512, 1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, 1),
            # nn.BatchNorm2d(256)

            # nn.ReLU(inplace=True),
            # nn.Conv2d(1024, 256, 1)
            # # nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )

    def forward(self, x_0, x0_0):
        # t = torch.cat((x, y, z), 0)
        # x0 is residual
        response_0 = self.et_0(x_0)
        # response_0 += x0_0

        return response_0

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == '__main__':

    # network test
    net = ET()
    net.eval()



