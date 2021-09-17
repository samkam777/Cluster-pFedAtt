# https://blog.csdn.net/guyuealian/article/details/88426648

import torch 
 
class Regularization(torch.nn.Module):
    def __init__(self,model,outer_weight_model,weight_decay,p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list, outer_weight_list = self.get_weight(model, outer_weight_model)
        #self.weight_info(self.weight_list, outer_weight_list, self.model)
 
    def to(self,device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device=device
        super().to(device)
        return self
 
    def forward(self, model, outer_weight_model):
        self.weight_list, outer_weight_list = self.get_weight(model, outer_weight_model)   # 获得最新的权重(当前模型的),即公式里的selta
        reg_loss = self.regularization_loss(self.weight_list, outer_weight_list, self.weight_decay, p=self.p)
        return reg_loss
 
    def get_weight(self, model, outer_weight_model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        # weight_list = []
        # for name, param in model.named_parameters():
        #     if 'weight' in name:
        #         weight = (name, param)
        #        weight_list.append(weight)
        # return weight_list
        weight_list = []            # selta
        outer_weight_list = []      # w
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = param
                weight_list.append(weight)
        
        for name, param in outer_weight_model.named_parameters():
            if 'weight' in name:
                weight = param
                outer_weight_list.append(weight)

        return weight_list, outer_weight_list

 
    def regularization_loss(self, weight_list, outer_weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss=[]
        # 一次循环是一个网络层
        for inner_w, outer_w in zip(weight_list, outer_weight_list):
            #print('inner_w: {}\t'.format(inner_w-outer_w))
            #print('len of inner_w: {}\t'.format(len(inner_w-outer_w)))
            l2_reg = torch.norm(inner_w.data - outer_w.data, p=p)
            l2_reg = l2_reg / len(inner_w-outer_w)
            #print('mean of l2_reg: {}\t'.format(l2_reg))
            #l2_reg = torch.norm(inner_w.data, p=p)
            #print('inner_w : {}\t'.format(len(inner_w)))
            #print('outer_w : {}\t'.format(len(outer_w)))
            #l2_reg = (inner_w.data - outer_w.data)**2
            #print('l2_reg: {}\t'.format(l2_reg))
            reg_loss.append(l2_reg)
        #print('regloss finish!')

        reg_loss_mean = sum(reg_loss) / len(reg_loss)

        reg_loss_mean=weight_decay*reg_loss_mean
        #print('reg_loss_mean: {}\t'.format(reg_loss_mean))
        return reg_loss_mean
 
    def weight_info(self,weight_list, outer_weight_list, model):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for inner_w, outer_w in zip(weight_list, outer_weight_list):
            l2_reg = torch.norm(inner_w.data - outer_w.data, p=2)
            print('l2_reg: {}\t'.format(l2_reg))

        #for (name, param) in model.named_parameters():
        #    if 'weight' in name:
        #        weight = (name, param)
        #    print('inner_w : {}\t'.format(name))



        print("---------------------------------------------------")




