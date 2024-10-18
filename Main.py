import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model import *
from init_parameter import *
from dataloader import *
from pretrain2 import *
from util import *
from contrastive_loss import *
import sys
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



class ModelManager:
    
    def __init__(self, args, data, pretrained_model=None):
        set_seed(args.seed)
        
        #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = len(data.all_label_list)
        #self.num_labels = 19
        if pretrained_model is None:
            
            pretrained_model = BertForModel.from_pretrained(args.bert_model, self.num_labels)
            pretrained_model.to(self.device)
            
            root_path = "pretrain_models"
            pretrain_dir = os.path.join(root_path, args.pretrain_dir)
            if os.path.exists(pretrain_dir):
                pretrained_model = self.restore_model(args, pretrained_model)
        self.pretrained_model = pretrained_model

        #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id           
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #if torch.cuda.device_count() > 1:
            #self.pretrained_model = torch.nn.DataParallel(self.pretrained_model)
        self.pretrained_model.to(self.device)
        
        #self.num_labels = 23
        #1
        #print("K:",self.predict_k(args, data) )
        #with open("results.txt", 'a') as file:
        #    file.write(str(self.predict_k(args, data)))
        self.known_data=data.known_label_list
        self.test_unlabeled_examples=data.test_examples
        #self.unknown_label_list=data.unknown_label_list

        self.lambda_cluster=args.lambda_cluster

        print("novel_num_label",self.num_labels)
        """
        if args.dataset=="clinc":
            self.model = BertForModel.from_pretrained(args.bert_model, num_labels = self.num_labels)
        else:
            self.model=self.pretrained_model
        """
        self.model=self.pretrained_model
        #self.model = BertForModel.from_pretrained(args.bert_model, num_labels = self.num_labels)
        #BertForModel.from_pretrained(args.bert_model, num_labels = data.n_unknown_cls)
        #if torch.cuda.device_count() > 1:
         
        #   self.model = torch.nn.DataParallel(torch.device("cuda:0"))
        if args.pretrain:
            self.load_pretrained_model(args)

        if args.freeze_bert_parameters:
            self.freeze_parameters(self.model)

        #total = sum([param.nelement() for param in self.model.parameters() if param.requires_grad])
        #print("Number of parameter: % .2fM" % (total / 1e6))
        #exit()
        
        self.model.to(self.device)
        #print(self.model)

        num_train_examples = len(data.train_unlabeled_examples)+ len(data.train_labeled_examples)
        self.num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs
        
        self.optimizer = self.get_optimizer(args)

        self.criterion_Maskinstance = MaskInstanceLoss(args.train_batch_size, args.instance_temperature, self.device).to(
            self.device)
        self.criterion_instance = InstanceLoss(args.train_batch_size, args.instance_temperature, self.device).to(
            self.device)
        self.criterion_instance_center = InstanceLoss(self.num_labels, args.instance_temperature, self.device).to(
            self.device)
        self.criterion_cluster = ClusterLoss(self.num_labels, args.cluster_temperature, self.device).to(
            self.device)

        self.best_eval_score = 0
        self.centroids = None
        self.training_SC_epochs = {}

        self.test_results = None
        self.predictions = None
        self.true_labels = None
        self.loss=[]
        #print("K:",self.predict_k(args, data) )
        #with open("results.txt", 'a') as file:
        #    file.write(str(self.predict_k(args, data)))

    def get_features_labels(self, dataloader, model, args):

        model.eval()
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                feature = model(batch, mode = 'feature-extract')

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels

    def predict_k(self, args, data):

        feats, _ = self.get_features_labels(data.train_semi_dataloader, self.pretrained_model, args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters = data.num_labels).fit(feats)
        y_pred = km.labels_

        pred_label_list = np.unique(y_pred)
        drop_out = len(feats) / data.num_labels
        print('drop',drop_out)

        cnt = 0
        for label in pred_label_list:
            num = len(y_pred[y_pred == label])
            if num < drop_out:
                cnt += 1

        num_labels = len(pred_label_list) - cnt
        print('pred_num',num_labels)

        return num_labels


    def get_optimizer(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                         lr = args.lr,
                         warmup = args.warmup_proportion,
                         t_total = self.num_train_optimization_steps)
        return optimizer

    def pca_visualization(self,x,y,predicted):
        label_list=[0,1,2,3,4,5,6,7,8,9]
        path = args.save_results_path
        pca_visualization(x, y, label_list, os.path.join(path, "pca_test.png"))
        pca_visualization(x, predicted, label_list, os.path.join(path, "pca_test_2.png"))

    def tsne_visualization(self,x,y,predicted):
        label_list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        path = args.save_results_path
        TSNE_visualization(x, y, label_list, os.path.join(path, "pca_test_b2.png"))
        TSNE_visualization(x, predicted, label_list, os.path.join(path, "pca_test_2_b2.png"))

    def tsne_visualization_2(self,x,y,predicted, epoch=100):
        label_list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        path = args.save_results_path
        TSNE_visualization(x, y, label_list, os.path.join(path, "PLPCL.png"))

    def evaluation_origin(self, args, data):

        self.model.eval()
        eval_dataloader = data.test_dataloader
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)

        step=0
        for batch in tqdm(eval_dataloader, desc="evaluation"):
            step+=1
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits, feat = self.model.forward_cluster(batch)
                total_features = torch.cat((total_features, feat))
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = total_logits.max(dim=1)
        x_feats = total_features.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        #print(y_pred)
        #print(y_true)
       
        results = clustering_score(y_true, y_pred)
        score = metrics.silhouette_score(x_feats, y_pred)
        results["SC"] = score
        print(results)

        ind, _ = hungray_aligment(y_true, y_pred)
        map_ = {i[0]: i[1] for i in ind}
        y_pred_aligned = np.array([map_[idx] for idx in y_pred])

        

        cm = confusion_matrix(y_true, y_pred_aligned)
        y_true_1=[]
        for i in y_true:
            y_true_1.append(i)
        y_pred_aligned_1=[]
        for i in y_pred_aligned:
            y_pred_aligned_1.append(i)
        #print(self.unknown_label_list)
        #print(y_true_1)
        #print(y_pred_aligned_1)
        #print(cm)
        

        """
        wrong=[]
        print(self.test_unlabeled_examples[0])
        
        for i in range(len(y_pred)):
            if y_pred_aligned[i]!=y_true[i]:
                text_a,label=self.test_unlabeled_examples[i]
                if_know=1 if label in self.known_data else 0
                wrong.append([text_a,label,y_true[i],y_pred_aligned[i],if_know])
        self.save_wrong_results(wrong,args)
        """
        self.test_results = results

        return results
    def evaluation_2(self, args, data):
        self.model.eval()
        eval_dataloader = data.test_unlabeled_dataloader
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)

        step = 0
        for batch in tqdm(eval_dataloader, desc="evaluation"):
            step += 1
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits, feat = self.model.forward_cluster(batch)
                total_features = torch.cat((total_features, feat))
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = total_logits.max(dim=1)
        x_feats = total_features.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        results = clustering_score(y_true, y_pred)
        print(results)

        ind, _ = hungray_aligment(y_true, y_pred)
        map_ = {i[0]: i[1] for i in ind}
        y_pred_aligned = np.array([map_[idx] for idx in y_pred])

        cm = confusion_matrix(y_true, y_pred_aligned)
        print(cm)

    def evaluation(self, args, data):

        self.model.eval()
        eval_dataloader = data.test_dataloader
        eval_IND_dataloader=data.test_IND_dataloader
        eval_OOD_dataloader=data.test_OOD_dataloader
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)
        IND_features = torch.empty((0, args.feat_dim)).to(self.device)
        IND_labels = torch.empty(0, dtype=torch.long).to(self.device)
        IND_logits = torch.empty((0, self.num_labels)).to(self.device)
        OOD_features = torch.empty((0, args.feat_dim)).to(self.device)
        OOD_labels = torch.empty(0, dtype=torch.long).to(self.device)
        OOD_logits = torch.empty((0, self.num_labels)).to(self.device)
        #OOD_logits = torch.empty((0, self.num_labels-len(self.known_data))).to(self.device)

        step=0
        for batch in tqdm(eval_dataloader, desc="evaluation"):
            step+=1
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits, feat = self.model.forward_cluster(batch)
                total_features = torch.cat((total_features, feat))
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))
        for batch in tqdm(eval_IND_dataloader, desc="evaluation"):
            step+=1
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits, feat = self.model.forward_cluster(batch)
                IND_features = torch.cat((IND_features, feat))
                IND_labels = torch.cat((IND_labels, label_ids))
                IND_logits = torch.cat((IND_logits, logits))
        for batch in tqdm(eval_OOD_dataloader, desc="evaluation"):
            step+=1
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits, feat = self.model.forward_cluster(batch)
                OOD_features = torch.cat((OOD_features, feat))
                OOD_labels = torch.cat((OOD_labels, label_ids))
                OOD_logits = torch.cat((OOD_logits, logits))

        total_probs, total_preds = total_logits.max(dim=1)
        IND_probs, IND_preds = IND_logits.max(dim=1)
        OOD_probs, OOD_preds = OOD_logits.max(dim=1)
        x_feats = total_features.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        IND_feats = IND_features.cpu().numpy()
        IND_pred = IND_preds.cpu().numpy()
        IND_true = IND_labels.cpu().numpy()
        OOD_feats = OOD_features.cpu().numpy()
        OOD_pred = OOD_preds.cpu().numpy()
        OOD_true = OOD_labels.cpu().numpy()
        #print(y_pred)
        #print(y_true)
       
        #results = {'All':clustering_accuracy_score(y_true, y_pred),'IND':clustering_accuracy_score(IND_true, IND_pred),"OOD":clustering_accuracy_score(OOD_true, OOD_pred)}
        #results = clustering_score(y_true, y_pred, data.known_lab)
        results = clustering_score(y_true, y_pred)
        score = metrics.silhouette_score(x_feats, y_pred)
        results["SC"] = score
        #print(results)

        ind, _ = hungray_aligment(y_true, y_pred)
        map_ = {i[0]: i[1] for i in ind}
        y_pred_aligned = np.array([map_[idx] for idx in y_pred])

        

        cm = confusion_matrix(y_true, y_pred_aligned)
        y_true_1=[]
        for i in y_true:
            y_true_1.append(i)
        y_pred_aligned_1=[]
        for i in y_pred_aligned:
            y_pred_aligned_1.append(i)
        #print(self.unknown_label_list)
        print(y_true_1)
        print(y_pred_aligned_1)
        print(cm)
        

        
        wrong=[]
        print(self.test_unlabeled_examples[0])
        
        for i in range(len(y_pred)):
            if y_pred_aligned[i]!=y_true[i]:
                text_a,label=self.test_unlabeled_examples[i].text_a,self.test_unlabeled_examples[i].label
                if_know=1 if label in self.known_data else 0
                wrong.append([text_a,label,y_true[i],y_pred_aligned[i],if_know])
        #self.save_wrong_results(wrong,args)
        
        self.test_results = results
        #np.savetxt('./outputs_check/cm.txt', cm)

        return results

    def visualize_training(self, args, data):
        self.model.eval()
        eval_dataloader = data.train_semi_dataloader
        total_features = torch.empty((0, 768)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)

        step = 0
        for batch in tqdm(eval_dataloader, desc="evaluation"):
            step += 1
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits, feat = self.model.forward_cluster(batch)
                total_features = torch.cat((total_features, feat))
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = total_logits.max(dim=1)
        total_features = normalize(total_features, dim=1)
        x_feats = total_features.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        # acc = round(accuracy_score(y_true, y_pred) * 100, 2)

        results = clustering_score(y_true, y_pred,data.known_lab)
        print(results)
        self.train_results = results

        # self.pca_visualization(x_feats, y_true, y_pred)
        self.tsne_visualization_2(x_feats, y_true, y_pred)

    def read_tsv(self,input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines
    def get_text(self,i):
            data_dir = os.path.join(args.data_dir, args.dataset)
            lines= self.read_tsv(os.path.join(data_dir, "test.tsv"))

            examples = []
            """
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                if len(line) != 2:
                    continue
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                label = line[1]
            """
            if len(lines[i])==2:
                return lines[i][0],lines[i][1]
            else:
                return "n","n"
    def save_wrong_results(self, values,args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        #var = [args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.cluster_num_factor, args.seed, self.num_labels]
        #names = ['text', 'label', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor','seed', 'K']
        #vars_dict = {k:v for k,v in zip(names, var) }
        #results = dict(self.test_results,**vars_dict)
        #keys = list(results.keys())
        #values = list(results.values())
        keys=["text","label","label_true","label_wrong","if know"] 
        
        file_name = 'wrong_results_PLPCL_0.7_banking.csv'
        results_path = os.path.join(args.save_results_path, file_name)
        
        if not os.path.exists(results_path):
            df1 = pd.DataFrame(values,columns = keys)
            df1.to_csv(results_path,index=False)
        else:
            df1 = pd.read_csv(results_path)
            #print(df1)
            new = pd.DataFrame(values,columns=keys)
            #for i in range(1,len(values)):
                #new.append(values[i],index=1)
            df1 = df1.append(new,ignore_index=True)
            #print(df1)
            df1.to_csv(results_path,index=False)
        data_diagram = pd.read_csv(results_path)
    
    def save_loss_results(self, values,args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        #var = [args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.cluster_num_factor, args.seed, self.num_labels]
        #names = ['text', 'label', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor','seed', 'K']
        #vars_dict = {k:v for k,v in zip(names, var) }
        #results = dict(self.test_results,**vars_dict)
        #keys = list(results.keys())
        #values = list(results.values())
        keys=["loss","loss_cluster","loss_instance"] 
        
        file_name = 'PLPCL_loss.csv'
        results_path = os.path.join(args.save_results_path, file_name)
        
        if not os.path.exists(results_path):
            df1 = pd.DataFrame(values,columns = keys)
            df1.to_csv(results_path,index=False)
        else:
            df1 = pd.read_csv(results_path)
            #print(df1)
            new = pd.DataFrame(values,columns=keys)
            #for i in range(1,len(values)):
                #new.append(values[i],index=1)
            df1 = df1.append(new,ignore_index=True)
            #print(df1)
            df1.to_csv(results_path,index=False)
        data_diagram = pd.read_csv(results_path)
    def eval(self, args, data, type):
        self.model.eval()
        eval_dataloader = data.eval_dataloader
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)

        step = 0
        for batch in tqdm(eval_dataloader, desc="evaluation"):
            step += 1
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits, feat = self.model.forward_cluster(batch)
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))
                #print(logits)
                #print(logits.size())
                total_features = torch.cat((total_features, feat))

        total_probs, total_preds = total_logits.max(dim=1)
        x_feats = total_features.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        #print(x_feats)
        #print(y_pred) 
        score = metrics.silhouette_score(x_feats,y_pred)
        # acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        #results = clustering_score(y_true, y_pred,data.known_lab)
        results = clustering_score(y_true, y_pred)
        #print(results)
        #self.test_results = results
        if type == 1:
            return results["ARI"]
        else:
            return score

    def training_process_eval(self, args, data, epoch):
        self.model.eval()
        eval_dataloader = data.train_semi_dataloader
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)

        step = 0
        for batch in tqdm(eval_dataloader, desc="evaluation"):
            step += 1
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits, feat = self.model.forward_cluster(batch)
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))
                total_features = torch.cat((total_features, feat))

        total_probs, total_preds = total_logits.max(dim=1)
        total_features = normalize(total_features, dim=1)
        x_feats = total_features.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        results = clustering_score(y_true, y_pred)
        score = metrics.silhouette_score(x_feats, y_pred)
        #score = results["NMI"]
        # acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        self.training_SC_epochs["epoch:" + str(epoch)] = score

        #self.tsne_visualization_2(x_feats, y_true, y_pred, epoch)

        return score


    def train(self, args, data):

        best_score = 0
        best_model = None
        wait = 0
        e_step = 0

        #SC_score = self.training_process_eval(args, data, e_step)
        #e_step += 1
        #print(SC_score)

        train_dataloader_1 = data.train_semi_dataloader
        
        #contrastive clustering
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            loss = 0
            self.model.train()
            step = 0
            loss_epoch = 0
            loss_cluster_center=0
            
            for step, batch in enumerate(tqdm(data.train_dataloader, desc="Pseudo-Training")):
                #print("bacth:",len(batch),batch)
                #batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, if_known = batch
                #print("input_ids",input_ids)
                batch=input_ids, input_mask, segment_ids, label_ids
                batch= tuple(t.to(self.device) for t in batch)
                z_i, z_j, c_i, c_j = self.model(batch, mode='contrastive-clustering')
                max_indices_i, max_indices_j= torch.argmax(c_i, axis=1),torch.argmax(c_j, axis=1)
                ifknown = []
                for i in range(len(if_known)):
                    if if_known[i] == 1 or c_i[i][max_indices_i[i]] > self.lambda_cluster:
                        ifknown.append(1)
                        if if_known[i] != 1:
                            label_ids[i]=max_indices_i[i]
                    else:
                        ifknown.append(0)
                #mask=torch.tensor([True if if_known[i] == 0 else False for i in range(len(batch)) ])
                #mask = mask.unsqueeze(1)
                #input_ids1 =input_ids[mask]
                input_ids1= torch.tensor([input_ids[i].tolist()  for i in range(len(if_known)) if ifknown[i] == 1])
                input_ids2= torch.tensor([input_ids[i].tolist()  for i in range(len(if_known)) if ifknown[i] == 0])
                input_mask1 = torch.tensor([input_mask[i].tolist() for i in range(len(if_known)) if ifknown[i] == 1])
                input_mask2= torch.tensor([input_mask[i].tolist()  for i in range(len(if_known)) if ifknown[i] == 0])
                segment_ids1 = torch.tensor([segment_ids[i].tolist() for i in range(len(if_known)) if ifknown[i] == 1])
                segment_ids2= torch.tensor([segment_ids[i].tolist()  for i in range(len(if_known)) if ifknown[i] == 0])
                label_ids1 = torch.tensor([label_ids[i].tolist() for i in range(len(if_known)) if ifknown[i] == 1])
                label_ids2= torch.tensor([label_ids[i].tolist()  for i in range(len(if_known)) if ifknown[i] == 0])
                
                
                if len(input_ids1)==0 or len(input_ids1)==1:
                    #z_i, z_j, c_i, c_j = self.model(batch, mode='contrastive-clustering')

                    max_indices_i, max_indices_j= torch.argmax(c_i, axis=1),torch.argmax(c_j, axis=1)
                    one_hot_i,one_hot_j = torch.eye(c_i.shape[1])[max_indices_i].to(self.device),torch.eye(c_j.shape[1])[max_indices_j].to(self.device)
                    c_i1,c_j1 = one_hot_i,one_hot_j
                    
                    zc_i,zc_j=torch.mm(c_i.t(),z_i),torch.mm(c_j.t(),z_j)
                    zc_i,zc_j=torch.nn.functional.normalize(zc_i, p=2, dim=0),torch.nn.functional.normalize(zc_j, p=2, dim=1)
                    loss_cluster_center=self.criterion_instance_center(zc_i,zc_j)

                    loss_instance = self.criterion_instance(z_i, z_j)
                    #loss_instance = self.criterion_Maskinstance(z_i, z_j, c_i, c_j)
                    #loss_instance = self.KNN_Instance(z_i, z_j, c_i, c_j)
                    loss_cluster = self.criterion_cluster(c_i, c_j)
                    loss = loss_instance + loss_cluster+loss_cluster_center
                    '''
                    loss_step = loss_instance + loss_cluster

                    if step%2 == 0:
                        loss = 0
                        loss = loss + loss_step
                        continue
                    else:
                        loss = loss + loss_step
                    '''

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    step+=1
                    #print(f"Step [{step}/{len(train_dataloader_1)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
                    loss_epoch += loss.item()
                    continue
                if len(input_ids2)==0 or len(input_ids2)==1:
                    max_indices_i, max_indices_j= torch.argmax(c_i, axis=1),torch.argmax(c_j, axis=1)
                    one_hot_i,one_hot_j = torch.eye(c_i.shape[1])[max_indices_i].to(self.device),torch.eye(c_j.shape[1])[max_indices_j].to(self.device)
                    c_i1,c_j1 = one_hot_i,one_hot_j
                    
                    zc_i,zc_j=torch.mm(c_i1.t(),z_i),torch.mm(c_j1.t(),z_j)
                    zc_i,zc_j=torch.nn.functional.normalize(zc_i, p=2, dim=0),torch.nn.functional.normalize(zc_j, p=2, dim=1)
                    loss_cluster_center=self.criterion_instance_center(zc_i,zc_j)

                    #loss_instance = self.criterion_instance(z_i, z_j)
                    #loss_instance = self.criterion_Maskinstance(z_i, z_j, c_i, c_j)
                    #loss_instance = self.KNN_Instance(z_i, z_j, c_i, c_j)
                    #loss_cluster = self.criterion_cluster(c_i, c_j)
                    loss = self.model(batch, mode = "pre-trained")+loss_cluster_center
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    step+=1
                    #print(f"Step [{step}/{len(train_dataloader_1)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
                    loss_epoch += loss.item()
                    continue
                
                batch1=input_ids1, input_mask1, segment_ids1, label_ids1
                batch2=input_ids2, input_mask2, segment_ids2, label_ids2
                #print(len(input_ids1))
                #print(len(input_ids2))
                #print(batch1)
                #print("input_ids",input_ids2)



                #print("batch1:",len(batch2),batch2)
                #batch1=input_ids1, input_mask1, segment_ids1, label_ids1 
                #batch2=input_ids2, input_mask2, segment_ids2, label_ids2
                #print("batch1:",len(batch2),batch2)
                batch2= tuple(t.to(self.device) for t in batch2)

                z_i, z_j, c_i, c_j = self.model(batch2, mode='contrastive-clustering')
                batch1= tuple(t.to(self.device) for t in batch1)
                z_i1, z_j1, c_i1, c_j1 = self.model(batch1, mode='contrastive-clustering')
                
                """
                print("c_i",c_i)
                print("c_i.size",c_i.shape)
                print("c_j",c_j)
                print("c_j.size",c_j.shape)
                print("z_i",z_i)
                print("z_i.size",z_i.shape)
                print("z_j",z_j)
                print("z_j.size",z_j.shape)
                """
                zc_i,zc_j=torch.mm(c_i.t(),z_i),torch.mm(c_j.t(),z_j)
                
                max_indices_i, max_indices_j= torch.argmax(c_i1, axis=1),torch.argmax(c_j1, axis=1)
                one_hot_i,one_hot_j = torch.eye(c_i1.shape[1])[max_indices_i].to(self.device),torch.eye(c_j1.shape[1])[max_indices_j].to(self.device)
                c_i1,c_j1 = one_hot_i,one_hot_j
                
                zc_i1,zc_j1=torch.mm(c_i1.t(),z_i1),torch.mm(c_j1.t(),z_j1)
                zc_i,zc_j=zc_i+zc_i1,zc_j+zc_j1
                zc_i,zc_j=torch.nn.functional.normalize(zc_i, p=2, dim=0),torch.nn.functional.normalize(zc_j, p=2, dim=1)
                if step==0:
                    zc_i_all,zc_j_all=zc_i,zc_j
                #zc_i_all,zc_j_all=torch.nn.functional.normalize(zc_i_all*step+zc_i, p=2, dim=0),torch.nn.functional.normalize(zc_j_all*step+zc_j, p=2, dim=0)

                loss_instance = self.criterion_instance(z_i, z_j)
                
                #loss_instance = self.criterion_Maskinstance(z_i, z_j, c_i, c_j)
                #loss_instance = self.KNN_Instance(z_i, z_j, c_i, c_j)
                loss_cluster = self.criterion_cluster(c_i, c_j)
                #loss = loss_instance + self.lambda_cluster*loss_cluster+loss_cluster_center
                
                #print(loss_cluster_center)
                loss_cluster_center=self.criterion_instance_center(zc_i,zc_j)
                loss = loss_instance + loss_cluster+self.model(batch1, mode = "pre-trained")+loss_cluster_center
                #self.loss.append([loss.item(),loss_cluster.item(),loss_instance.item(),loss_cluster_center.item()])
                self.loss.append([loss.item(),loss_cluster.item(),loss_instance.item()])
                ''' 
                loss_step = loss_instance + loss_cluster

                if step%2 == 0:
                    loss = 0
                    loss = loss + loss_step
                    continue
                else:
                    loss = loss + loss_step
                '''

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                step+=1
                #print(f"Step [{step}/{len(train_dataloader_1)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}\t loss_cluster_center: {loss_cluster_center.item()}")
                #print(f"Step [{step}/{len(train_dataloader_1)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
                loss_epoch += loss.item()
            #print(f"Epoch [{epoch}/{args.num_train_epochs}]\t Loss: {loss_epoch / len(train_dataloader_1)}")

            #SC_score = self.training_process_eval(args, data, e_step)
            #e_step += 1
            #print(SC_score)

            eval_acc = self.eval(args, data, 0)
            #print(eval_acc)
            if eval_acc > best_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                best_score = eval_acc
            else:
                wait += 1
                if wait >= args.wait_patient:
                    self.model = best_model
                    break
        #print("K:",self.predict_k(args, data) )
        self.save_loss_results(self.loss,args)
        self.model = best_model
        if args.save_model:
            self.save_model(args)

    def save_model(self, args):
        root_path = "PLPCL_models"
        pretrain_dir = os.path.join(root_path, "PLPCL_0.7")
        if not os.path.exists(pretrain_dir):
            os.makedirs(pretrain_dir)
        self.save_model = self.model.module if hasattr(self.model, 'module') else self.model
        model_file = os.path.join(pretrain_dir, WEIGHTS_NAME)
        model_config_file = os.path.join(pretrain_dir, CONFIG_NAME)
        torch.save(self.save_model.state_dict(), model_file)
        with open(model_config_file, "w") as f:
            f.write(self.save_model.config.to_json_string())

    def load_pretrained_model(self, args):
        pretrained_dict = self.pretrained_model.state_dict()
        classifier_params = ['classifier.weight', 'classifier.bias', 'cluster_projector.2.weight', 'cluster_projector.2.bias']
        #pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        pretrained_dict =  {k: v for k, v in pretrained_dict.items()}
        self.model.load_state_dict(pretrained_dict, strict=False)
        

    def restore_model(self, args, model):
        root_path = "pretrain_models"
        pretrain_dir = os.path.join(root_path, args.pretrain_dir)
        output_model_file = os.path.join(pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model
    
    def freeze_parameters(self,model):
        for name, param in model.bert.named_parameters():  
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.method, args.known_cls_ratio, args.cluster_num_factor, args.seed, args.train_batch_size, args.lr, self.num_labels,self.lambda_cluster,args.labeled_ratio]
        names = ['dataset', 'method', 'known_cls_ratio', 'cluster_num_factor','seed', 'train_batch_size', 'learning_rate', 'K','lambda_cluster','labeled_ratio']
        vars_dict = {k:v for k,v in zip(names, var) }
        results = dict(self.test_results,**vars_dict)
        keys = list(results.keys())
        values = list(results.values())
        
        file_name = 'reselts_check_v7.csv'
        results_path = os.path.join(args.save_results_path, file_name)
        
        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori,columns = keys)
            df1.to_csv(results_path,index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results,index=[1])
            df1 = df1.append(new,ignore_index=True)
            df1.to_csv(results_path,index=False)
        data_diagram = pd.read_csv(results_path)
        
        print('test_results', data_diagram)
        #self.save_training_process(args)

    def save_training_process(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        results = dict(self.training_SC_epochs)
        keys = list(results.keys())
        values = list(results.values())

        file_name = 'results_analysis_V2_100_trainigEpoch.csv'
        results_path = os.path.join(args.save_results_path, file_name)

        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results, index=[1])
            df1 = df1.append(new, ignore_index=True)
            df1.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)

        print('training_process_dynamic:', data_diagram)

if __name__ == '__main__':
    #print("11111111111111")
    print('Data and Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()
    print("gpu_id",args.gpu_id)
    data = Data(args)
    #args.dataset="error_"+args.dataset
    #error_data=Data(args)


    if args.pretrain:
        if args.method == "PLPCL":
            torch.backends.cudnn.enabled=False
            print('Pre-training begin...')
            #print("gpu_id",args.gpu_id)
            manager_p = PretrainModelManager(args,data)
            
            #manager_p.train(args, error_data)
            #args.dataset="banking"
            manager_p.train(args, data)
            print('Pre-training finished!')
            torch.cuda.empty_cache()
            #manager_p.load_models(args)
            #manager_p.analysis(args, data)
            #manager_p.save_results(args) 
            #exit()
            #manager_p.evaluation(args, data)
            #exit()

            manager = ModelManager(args, data)
            print('Training begin...')
            manager.train(args, data)
            print('Training finished!')
            

            print('Evaluation begin...')
            manager.evaluation(args, data)
            print('Evaluation finished!')
            #manager.visualize_training(args, data)

            manager.save_results(args)

