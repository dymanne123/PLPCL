from util import *
from model import *
from dataloader import *
import re
from keras.utils import to_categorical
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys
# Evaluation
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#from tensorboard_logger import Logger

class PretrainModelManager:
    
    def __init__(self, args, data):
        set_seed(args.seed)

        #self.model = BertForModel.from_pretrained(args.bert_model, num_labels = data.n_known_cls)
        self.model = BertForModel.from_pretrained(args.bert_model, num_labels = len(data.all_label_list))
        #self.model = BertForModel.from_pretrained(args.bert_model, num_labels = 19)
        #print(self.model) 
        if args.freeze_bert_parameters:
            self.freeze_parameters(self.model) 
        root_path = "./"
        pretrain_dir = os.path.join(root_path, args.pretrain_dir)
        if os.path.exists(pretrain_dir):
            self.model = self.restore_model(args, self.model)
        
        #self.load_dpn_pretrained_model(args)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id           
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model.to(self.device) 
        self.num_labels = len(data.all_label_list)
        #self.num_labels =19
        #n_gpu = torch.cuda.device_count()
        #if n_gpu > 1:
        #    self.model = torch.nn.DataParallel(self.model)

        self.num_train_optimization_steps = int(len(data.train_labeled_examples) / args.pre_train_batch_size) * args.num_pretrain_epochs
        
        self.optimizer = self.get_optimizer(args)
        self.train_labeled_examples=data.train_labeled_examples_list
        self.test_unlabeled_examples=data.test_examples
        self.known_data=data.known_label_list
        self.best_eval_score = 0
        self.analysis_results = {}

    def load_dpn_pretrained_model(self, args):
        pretrained_dict = self.model.state_dict()
        classifier_params = ['classifier.weight', 'classifier.bias', 'cluster_projector.2.weight', 'cluster_projector.2.bias']
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)
    def restore_model(self, args, model):
        root_path = "./"
        pretrain_dir = os.path.join(root_path, args.dpn_pretrain_dir)
        output_model_file = os.path.join(pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model
    def load_models(self, args):
        print("loading models ....")
        self.model = self.restore_model_v2(args, self.model)

    def create_negative_dataset(self, data, args):
        negative_dataset = {}
        train_dataset = data.train_labeled_examples
        all_IND_data = data.get_embedding(train_dataset, data.known_label_list, args, "train")
        #print(all_IND_data)

        for line in all_IND_data:
            label = int(line["label_id"])
            inputs = line

            inputs.pop("label_id")
            if label not in negative_dataset.keys():
                negative_dataset[label] = [inputs]
            else:
                negative_dataset[label].append(inputs)

        #exit()
        return negative_dataset

    def generate_positive_sample(self, label: torch.Tensor):
        positive_num = self.positive_num

        # positive_num = 16
        positive_sample = []
        for index in range(label.shape[0]):
            input_label = int(label[index])
            positive_sample.extend(random.sample(self.negative_data[input_label], positive_num))

        return self.reshape_dict(positive_num, self.list_item_to_tensor(positive_sample))

    @staticmethod
    def list_item_to_tensor(inputs_list: List[Dict]):
        batch_list = {}
        for key, value in inputs_list[0].items():
            batch_list[key] = []
        for inputs in inputs_list:
            for key, value in inputs.items():
                batch_list[key].append(value)

        batch_tensor = {}
        for key, value in batch_list.items():
            batch_tensor[key] = torch.tensor(value)
        return batch_tensor

    def reshape_dict(self, sample_num, batch):
        
        for k, v in batch.items():
            shape = v.shape
            batch[k] = v.view([-1, sample_num, shape[-1]])
        return batch

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        return inputs
    

    def evaluation2(self, args, data):
        self.model.eval()
        train_dataloader = data.train_labeled_dataloader
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.n_known_cls)).to(self.device)

        for batch in tqdm(train_dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                feat, logits = self.model(batch, mode='eval')
                total_features = torch.cat((total_features, feat))
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        wrong=[]
        print(self.train_labeled_examples[0])
        """
        ind, _ = hungray_aligment(y_true, y_pred)
        map_ = {i[0]: i[1] for i in ind}
        y_pred_aligned = np.array([map_[idx] for idx in y_pred])
        """
        for i in range(len(y_pred)):
            if y_pred[i]!=y_true[i]:
                print(self.train_labeled_examples[i])
                text_a,label=self.train_labeled_examples[i][0],self.train_labeled_examples[i][-1]
                #if_know=1 if label in self.known_data else 0
                #wrong.append([text_a,label,y_true[i],y_pred_aligned[i],if_know])
                wrong.append([text_a,label])
        self.save_wrong_results(wrong,args,data)

        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        print("accuracy:",acc)

        return acc

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
    def get_text(self,args,i):
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
    def save_wrong_results(self, values,args,data):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        #var = [args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.cluster_num_factor, args.seed, self.num_labels]
        #names = ['text', 'label', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor','seed', 'K']
        #vars_dict = {k:v for k,v in zip(names, var) }
        #results = dict(self.test_results,**vars_dict)
        #keys = list(results.keys())
        #values = list(results.values())
        #keys=["text","label","label_true","label_wrong","if know"]
        keys=["text","label"] 
        
        file_name = 'error_banking.tsv'
        results_path = os.path.join(data.data_dir, file_name)
        
        if not os.path.exists(results_path):
            df1 = pd.DataFrame(values,columns = keys)
            df1.to_csv(results_path,index=False,sep="\t")
        else:
            df1 = pd.read_csv(results_path)
            #print(df1)
            new = pd.DataFrame(values,columns=keys)
            #for i in range(1,len(values)):
                #new.append(values[i],index=1)
            df1 = df1.append(new,ignore_index=True)
            #print(df1)
            df1.to_csv(results_path,index=False,sep="\t")
        data_diagram = pd.read_csv(results_path)

    def analysis(self, args, data):
        self.model.eval()
        test_dataloader = data.train_labeled_dataloader
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.n_known_cls)).to(self.device)

        for batch in tqdm(test_dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                feat, logits = self.model(batch, mode='eval')
                total_features = torch.cat((total_features, feat))
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        x_feats = total_features.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        min_d, max_d, mean_d, _ = intra_distance(x_feats, y_true, data.n_known_cls)
        self.analysis_results["intra_distance"] = mean_d
        min_d, max_d, mean_d, _ = inter_distance(x_feats, y_true, data.n_known_cls)
        self.analysis_results["inter_distance"] = mean_d
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        print("accuracy:", acc)

        return acc



    def eval(self, args, data):
        self.model.eval()
        total_features = torch.empty((0, 768)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)
        
        for batch in tqdm(data.eval_dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                feat, logits = self.model(batch, mode = 'eval')
                total_features = torch.cat((total_features, feat))
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim = 1)
        x_feats = total_features.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        acc = round(accuracy_score(y_true, y_pred) * 100, 2)

        return acc


    def train(self, args, data):  
 
        wait = 0
        best_model = self.model
        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(data.train_labeled_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                with torch.set_grad_enabled(True):
                    loss = self.model(batch, mode = "pre-trained")
                    loss.backward()
                    tr_loss += loss.item()
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
            
            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)
            
            #self.evaluation(args,data)
            #self.save_results(args)
            if args.dataset!="error_banking":
                eval_score = self.eval(args, data)
                print('eval_score',eval_score)
            
        
                if eval_score > self.best_eval_score:
                    best_model = copy.deepcopy(self.model)
                    wait = 0
                    self.best_eval_score = eval_score
                else:
                    wait += 1
                    if wait >= args.wait_patient:
                        break

        #print(self.evaluation(args,data))
        if args.dataset!="error_banking":
            self.model = best_model
        if args.save_model and args.dataset!="error_banking":
            self.save_model(args)

    def get_optimizer(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                         lr = args.lr_pre,
                         warmup = args.warmup_proportion,
                         t_total = self.num_train_optimization_steps)   
        return optimizer
    
    def save_model(self, args):
        root_path = "pretrain_models"
        pretrain_dir = os.path.join(root_path, args.pretrain_dir)
        if not os.path.exists(pretrain_dir):
            os.makedirs(pretrain_dir)
        self.save_model = self.model.module if hasattr(self.model, 'module') else self.model  
        model_file = os.path.join(pretrain_dir, WEIGHTS_NAME)
        model_config_file = os.path.join(pretrain_dir, CONFIG_NAME)
        torch.save(self.save_model.state_dict(), model_file)
        with open(model_config_file, "w") as f:
            f.write(self.save_model.config.to_json_string())
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
        print(y_pred)
        print(y_true)
       
        #results = {'All':clustering_accuracy_score(y_true, y_pred),'IND':clustering_accuracy_score(IND_true, IND_pred),"OOD":clustering_accuracy_score(OOD_true, OOD_pred)}
        results = clustering_score(y_true, y_pred, data.known_lab)
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
    def freeze_parameters(self,model):
        for name, param in model.bert.named_parameters():  
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def restore_model_v2(self, args, model):
        root_path = "pretrain_models"
        pretrain_dir = os.path.join(root_path, args.pretrain_dir)
        output_model_file = os.path.join(pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model

    def save_results2(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.cluster_num_factor, args.seed,
               args.train_batch_size, args.lr, 8]
        names = ['dataset', 'method', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor', 'seed',
                 'train_batch_size', 'learning_rate', 'K']
        vars_dict = {k: v for k, v in zip(names, var)}
        results = dict(self.analysis_results, **vars_dict)
        keys = list(results.keys())
        values = list(results.values())

        file_name = 'analysis_1.csv'
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

        print('test_results', data_diagram)
    
    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.method, args.known_cls_ratio, args.cluster_num_factor, args.seed, args.train_batch_size, args.lr, self.num_labels]
        names = ['dataset', 'method', 'known_cls_ratio', 'cluster_num_factor','seed', 'train_batch_size', 'learning_rate', 'K']
        vars_dict = {k:v for k,v in zip(names, var) }
        results = dict(self.test_results,**vars_dict)
        keys = list(results.keys())
        values = list(results.values())
        
        file_name = 'reselts_check_v5.csv'
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