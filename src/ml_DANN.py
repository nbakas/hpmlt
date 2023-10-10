

from import_libraries import *
from misc_functions import *

class torch_set_dataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data   
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    def __len__ (self):
        return len(self.X_data)
    

class MultipleRegression(nn.Module):
    def __init__(self, num_features, num_targets, neurons, dropout, layers):
        super(MultipleRegression, self).__init__()
        self.num_features = num_features
        self.num_targets = num_targets
        self.layers = layers
        self.dropout = dropout
        self.linear_layers = nn.ModuleList()
        # Add input layer
        self.linear_layers.append(nn.Linear(num_features, neurons))
        torch.nn.init.xavier_uniform_(self.linear_layers[-1].weight)
        self.linear_layers.append(nn.ReLU())
        self.linear_layers.append(nn.Dropout(p=dropout))
        # Add hidden layers
        for i in range(layers-1):
            self.linear_layers.append(nn.Linear(neurons, neurons))
            torch.nn.init.xavier_uniform_(self.linear_layers[-1].weight)
            self.linear_layers.append(nn.ReLU())
            self.linear_layers.append(nn.Dropout(p=dropout))
        # Add output layer
        self.linear_layers.append(nn.Linear(neurons, num_targets))
        torch.nn.init.xavier_uniform_(self.linear_layers[-1].weight)
    def forward(self, x):
        for layer in self.linear_layers:
            x = layer(x)
        return x


def plot_all_losses(all_losses_tr,all_losses_vl):
    # convert all_losses_vl to numpy array
    all_losses_vl = array(all_losses_vl)
    all_losses_tr = array(all_losses_tr)
    iso = argsort(all_losses_vl)[::-1]
    plt.plot(all_losses_tr[iso], label='Train Losses')
    plt.plot(all_losses_vl[iso], label='Validation Losses')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("all_losses_tuning.png")
    plt.close()


def plot_history(history,perc_of_epochs_to_plot,tag1,tag2,path_):
    nn = int(len(history['tr'])*(100-perc_of_epochs_to_plot)/100)
    # plot train and validation loss
    plt.plot(arange(1,len(history['tr'])+1)[nn:], array(history['tr'][nn:]), label=tag1, marker = 'o')
    plt.plot(arange(1,len(history['tr'])+1)[nn:], array(history['te'][nn:]), label=tag2, marker = 'o')
    plt.title('Pearson R')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(path_ + "loss_history_percentage_kept_"+str(perc_of_epochs_to_plot)+".png")
    plt.close()

def pred_DANN(XX):
    global model
    # Set the model to evaluation mode
    model.eval()
    # Convert the input data to a PyTorch tensor
    XXX = torch.tensor(XX).float()
    # XXX = XXX.to('cuda')
    # Use the model to make predictions on the input data
    with torch.no_grad():
        pred_tr = model(XXX)
    XXX=0
    model.train()
    # Convert the predicted values to a NumPy array
    pred_tr = pred_tr.cpu().numpy()
    return pred_tr.reshape(-1)

def pred_DANN_model_local(XX, model_local):
    # Set the model to evaluation mode
    model_local.eval()
    # Convert the input data to a PyTorch tensor
    XXX = torch.tensor(XX).float()
    # XXX = XXX.to('cuda')
    # Use the model_local to make predictions on the input data
    with torch.no_grad():
        pred_tr = model_local(XXX)
    XXX=0
    model_local.train()
    # Convert the predicted values to a NumPy array
    pred_tr = pred_tr.cpu().numpy()
    return pred_tr.reshape(-1)

def train_pytorch(XTR, YTR, XVL, YVL, XTE, YTE, params, epochs, path_, save_model=False):
    
    YTR = YTR.reshape(-1,1)
    YVL = YVL.reshape(-1,1)
    layers = params[0]; neurons = params[1]; 
    learning_rate = params[2]; dropout = params[3]; batch = int(params[4]); moment_um = params[5]
    
    train_dataset = torch_set_dataset(torch.from_numpy(XTR).float(), torch.from_numpy(YTR).float())

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=False)

    model_local = MultipleRegression(XTR.shape[1], YTR.shape[1], neurons, dropout, layers)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_local = nn.DataParallel(model_local)
    model_local.to(device)
    
    criterion = nn.MSELoss()
    
    optimizer = optim.NAdam(model_local.parameters(), lr=learning_rate, momentum_decay=moment_um)
    
    loss_stats = {'tr': [], "vl": [], "te": []}
    for e in range(1, epochs+1):
        model_local.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model_local(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch)
            train_loss.backward()
            optimizer.step() 
          
        loss_stats['tr'].append(corrcoef(pred_DANN_model_local(XTR,model_local),YTR.reshape(-1))[0,1])
        loss_stats['vl'].append(corrcoef(pred_DANN_model_local(XVL,model_local),YVL.reshape(-1))[0,1]) 
        loss_stats['te'].append(corrcoef(pred_DANN_model_local(XTE,model_local),YTE.reshape(-1))[0,1]) 

    if save_model:
        print("saving model...", flush=True)
        torch.save(model_local, path_ + "model.pth")
    
    return loss_stats, model_local


def train_dann_folds(i,combination,nof_folds,Xtr,ytr,Xte,yte,tr_inds,vl_inds,path_,
                     acc_tr_all,acc_vl_all,acc_te_all,epochs_all,nof_tune_epochs,blas_threads,IS_HPC):

    acc_tr = zeros(nof_tune_epochs); acc_vl = zeros(nof_tune_epochs); acc_te = zeros(nof_tune_epochs)
    for fold in range(nof_folds):
        
        if IS_HPC:
            with threadpool_limits(limits=int(blas_threads[i]), user_api='blas'):
                history, model_i = train_pytorch( Xtr[tr_inds[fold],:], ytr[tr_inds[fold]], 
                                                Xtr[vl_inds[fold],:], ytr[vl_inds[fold]], 
                                                Xte, yte, 
                                                combination, nof_tune_epochs, path_, False)
        else:
            history, model_i = train_pytorch( Xtr[tr_inds[fold],:], ytr[tr_inds[fold]], 
                                            Xtr[vl_inds[fold],:], ytr[vl_inds[fold]], 
                                            Xte, yte, 
                                            combination, nof_tune_epochs, path_, False)
            
        acc_tr += history['tr']
        acc_vl += history['vl']
        acc_te += history['te']
        acc_tr[isnan(acc_tr)] = 0
        acc_vl[isnan(acc_vl)] = 0
        acc_te[isnan(acc_te)] = 0
    acc_tr /= nof_folds; acc_vl /= nof_folds; acc_te /= nof_folds
    
    # the nBest = argmax(acc_vl) is computed and returned, 
    # providing information for the optimal number of epochs for the current set of hyper-parameters.
    nBest = argmax(acc_vl)
    acc_tr = acc_tr[nBest]; acc_vl = acc_vl[nBest]; acc_te = acc_te[nBest]; 
    acc_tr_all[i] = acc_tr
    acc_vl_all[i] = acc_vl
    acc_te_all[i] = acc_te
    epochs_all[i] = nBest+1
    
    sdt = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(sdt,i,"acc_tr",acc_tr,"acc_vl",acc_vl,"acc_te",acc_te,"opti_epochs",nBest+1,
          "blas_threads[i]=",blas_threads[i], flush=True)
    
    
def run_mult_proc_dann(combinations,nof_folds,Xtr,ytr,Xte,yte,tr_inds,vl_inds,path_,
                       nof_tune_epochs,blas_threads,IS_HPC):
    manager_tr = multiprocessing.Manager()
    acc_tr_all = manager_tr.dict()
    manager_vl = multiprocessing.Manager()
    acc_vl_all = manager_vl.dict()
    manager_te = multiprocessing.Manager()
    acc_te_all = manager_te.dict()
    manager_nBest = multiprocessing.Manager()
    epochs_all = manager_nBest.dict()
    jobs = []
    for i,combination in enumerate(combinations):
        p = multiprocessing.Process(target=train_dann_folds,
            args=(i,combination,nof_folds,Xtr,ytr,Xte,yte,tr_inds,vl_inds,path_,
                acc_tr_all,acc_vl_all,acc_te_all,epochs_all,nof_tune_epochs,blas_threads,IS_HPC))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    acc_tr_all = array(list(dict(sorted(acc_tr_all.items())).values()))
    acc_vl_all = array(list(dict(sorted(acc_vl_all.items())).values()))
    acc_te_all = array(list(dict(sorted(acc_te_all.items())).values()))
    epochs_all = array(list(dict(sorted(epochs_all.items())).values()))
    print("multiprocessing done",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3], flush=True)     
    return acc_tr_all, acc_vl_all, acc_te_all, epochs_all        

def plot_cv_history_dann(acc_tr_all,acc_vl_all,acc_te_all,path_,tag_):
    iso = argsort(acc_vl_all)
    ipositive = acc_vl_all[iso]>0
    plt.plot(acc_tr_all[iso][ipositive], 'x', label='Train')
    plt.plot(acc_vl_all[iso][ipositive], 'x', label='Validation')
    plt.plot(acc_te_all[iso][ipositive], 'x', label='Test')
    plt.legend()
    plt.xlabel('Model')
    plt.ylabel('R2 Score')
    plt.savefig(path_ + "DANN_tune_cv_history_" + tag_ + ".png")
    plt.close()
    
    iso = argsort(acc_vl_all)[int(0.5*len(acc_vl_all)):]
    ipositive = acc_vl_all[iso]>0
    plt.plot(acc_tr_all[iso][ipositive], 'x', label='Train')
    plt.plot(acc_vl_all[iso][ipositive], 'x', label='Validation')
    plt.plot(acc_te_all[iso][ipositive], 'x', label='Test')
    plt.legend()
    plt.xlabel('Model')
    plt.ylabel('R2 Score')
    plt.savefig(path_ + "DANN_tune_cv_history_50perc_" + tag_ + ".png")
    plt.close()
          
def do_DANN(Xtr,ytr,Xte,yte,features_names,target_name,IS_HPC,PERMUTE_TRAIN_TEST,LOGISTIC_REGR,ROOT_DIR,
                nof_1st_tune_models,nof_1st_tune_epochs,nof_2nd_tune_models,nof_2nd_tune_epochs,nof_final_blas_thr):
    global model###################################
    __method__ = "DANN"
    path_ = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep
    path_err = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep+"Error_Analysis"+os.sep
    path_sens = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep+"Sensitivity_Analysis"+os.sep
    try:
        t0=time()
        
        scaler = MinMaxScaler()
        scaler.fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xte = scaler.transform(Xte)
        obs = Xtr.shape[0]
        
        torch.manual_seed(0)
        seed(0)
        random.seed(0)
        
        #Tuning Parameters:
        
        # layers: It refers to the number of hidden layers in a neural network. Deep networks with multiple layers 
        # can capture complex patterns but are more prone to overfitting. Shallow networks with fewer layers may be 
        # simpler and less prone to overfitting but may struggle with complex tasks. The number of layers depends 
        # on the complexity of the problem, but a typical range is 1 to 10 layers.
        # Lower bound: 1. Upper bound: No strict upper bound, but typically around 10. Suggested values: 1 to 10
        layers = list(arange(1,11))#list(concatenate((arange(1,11),arange(20, 101, 10))))
        
        # neurons: It represents the number of neurons (also called units or nodes) in each hidden layer of a neural network. 
        # Higher numbers of neurons can capture more complex relationships, but they increase the computational cost and 
        # the risk of overfitting. The number of neurons in each layer is problem-dependent and typically determined through experimentation. 
        # There is no strict range or suggested values since it heavily depends on the specific problem.
        neurons = list(concatenate((arange(1,11),arange(20, 101, 10))))#,arange(200, 1001, 100)
        
        # learning_rate: It determines the step size used to update the weights of the neural network during training. 
        # A higher learning rate may lead to faster convergence but can also cause overshooting the optimal solution. 
        # A lower learning rate may improve the model's stability but increase training time. 
        # The default value is often set to 0.001, but the optimal value depends on the problem and the network architecture.
        learning_rate = concatenate((   np_round(linspace(0.0001,0.001,num=3), 10),
                                        np_round(linspace(0.001,0.01,num=3), 10),
                                        np_round(linspace(0.01,0.1,num=3), 10)))
        learning_rate = list(unique(learning_rate))
        
        # dropout: It is a regularization technique that randomly sets a fraction of the neurons to 0 during training to prevent overfitting. 
        # Dropout helps in reducing the interdependence between neurons and encourages the network to learn more robust features. 
        # The typical dropout rate ranges from 0.1 to 0.5.
        dropout =       concatenate((   np_round(linspace(0.001,0.01,num=3), 10),
                                        np_round(linspace(0.01,0.1,num=3), 10),
                                        np_round(linspace(0.1,0.5,num=3), 10)))
        dropout = list(unique(dropout))
        
        # batch size: It refers to the number of training samples propagated through the network before updating the weights. 
        # Larger batch sizes offer computational efficiency, but smaller batch sizes provide more stochasticity 
        # and can help escape local optima. The optimal batch size depends on the available memory, computational resources, 
        # and the dataset size. Common values range from 8 to 256.
        batch = 2**arange(1,31)
        while batch[-1]>obs/4:
            batch = batch[:-1]
            
        # momentum: It is a parameter that accelerates convergence by adding a fraction of the previous weight update 
        # to the current update. Momentum helps in navigating flat regions and escaping local minima during training. 
        # Typical values range from 0.9 to 0.99.
        moment_um = concatenate((   np_round(linspace(0.01,0.1,num=5), 10),
                                    np_round(linspace(0.1,0.999,num=5), 10)))
        moment_um = list(unique(moment_um))
        combinations = list(product(layers, neurons, learning_rate, dropout, batch, moment_um)) 
        print("combinations done",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3],"nof combinations=",len(combinations),flush=True)

        # randomly permute the combinations
        combinations = shuffle(combinations, random_state=0)
        if len(combinations)>nof_1st_tune_models:
            combinations = combinations[:nof_1st_tune_models]
        print("combinations edited",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3],"nof combinations=",len(combinations),flush=True)

        # split train to train and validation
        ___perc_cv___ = 0.8; nof_folds = 5; obs = len(ytr)
        tr_inds, vl_inds = split_tr_vl(obs,___perc_cv___,nof_folds,PERMUTE_TRAIN_TEST)
        
        
        if not IS_HPC:
            print("I do not run in parallel!",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3], flush=True)
            acc_tr_all = zeros(nof_1st_tune_models)-Inf; acc_vl_all = zeros(nof_1st_tune_models)-Inf; 
            acc_te_all = zeros(nof_1st_tune_models)-Inf; epochs_all = zeros(nof_1st_tune_models)
            bl_thr = 8*ones(nof_1st_tune_models, dtype=int)
            for i,combination in enumerate(combinations):
                try:
                    train_dann_folds(i,combination,nof_folds,Xtr,ytr,Xte,yte,tr_inds,vl_inds,path_,
                                    acc_tr_all,acc_vl_all,acc_te_all,epochs_all,nof_1st_tune_epochs,bl_thr,IS_HPC)
                except KeyboardInterrupt:
                    print('KeyboardInterrupt: Stopped by user')
                    break
            plot_cv_history_dann(acc_tr_all, acc_vl_all, acc_te_all, path_, "Initial")
            print("1 multiprocessing done",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3], flush=True)
            str_combs = "layers", "neurons", "learning_rate", "dropout", "batch", "moment_um"
            for istrcomb in range(len(combinations[0])):
                comb_i = [combo[istrcomb] for combo in combinations]
                print(str_combs[istrcomb],"min=", min(comb_i),"mean=", mean(comb_i),"max=", max(comb_i))
            print("epochs_all","min=", min(epochs_all),"mean=", mean(epochs_all),"max=", max(epochs_all))
                 
            iso = argsort(acc_vl_all)[-nof_2nd_tune_models:]
            combinations = [combinations[it] for it in iso]
            acc_tr_all = zeros(nof_2nd_tune_models)-Inf; acc_vl_all = zeros(nof_2nd_tune_models)-Inf; 
            acc_te_all = zeros(nof_2nd_tune_models)-Inf; epochs_all = zeros(nof_2nd_tune_models)
            bl_thr = 8*ones(nof_1st_tune_models, dtype=int)
            for i,combination in enumerate(combinations):
                try:
                    train_dann_folds(i,combination,nof_folds,Xtr,ytr,Xte,yte,tr_inds,vl_inds,path_,
                                    acc_tr_all,acc_vl_all,acc_te_all,epochs_all,nof_2nd_tune_epochs,bl_thr,IS_HPC)
                except KeyboardInterrupt:
                    print('KeyboardInterrupt: Stopped by user')
                    break
            print("2 multiprocessing done",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3], flush=True)
            for istrcomb in range(len(combinations[0])):
                comb_i = [combo[istrcomb] for combo in combinations]
                print(str_combs[istrcomb],"min=", min(comb_i),"mean=", mean(comb_i),"max=", max(comb_i))
            print("epochs_all","min=", min(epochs_all),"mean=", mean(epochs_all),"max=", max(epochs_all))
            
        else:
            print("I run in parallel!",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3], flush=True)
            avail_thr = os.cpu_count()-threading.active_count()
            bl_thr = ones(nof_1st_tune_models, dtype=int)
            while sum(bl_thr)<avail_thr:
                for ithr in range(len(bl_thr)):
                    if sum(bl_thr)<avail_thr:
                        bl_thr[ithr] += 1
            print("bl_thr",bl_thr,flush=True)
            acc_tr_all, acc_vl_all, acc_te_all, epochs_all = run_mult_proc_dann(combinations,
                    nof_folds,Xtr,ytr,Xte,yte,tr_inds,vl_inds,path_,nof_1st_tune_epochs,bl_thr,IS_HPC)
            plot_cv_history_dann(acc_tr_all, acc_vl_all, acc_te_all, path_, "Initial")
            print("1 multiprocessing done",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3], flush=True)
            str_combs = "layers", "neurons", "learning_rate", "dropout", "batch", "moment_um"
            for istrcomb in range(len(combinations[0])):
                comb_i = [combo[istrcomb] for combo in combinations]
                print(str_combs[istrcomb],"min=", min(comb_i),"mean=", mean(comb_i),"max=", max(comb_i))
            print("epochs_all","min=", min(epochs_all),"mean=", mean(epochs_all),"max=", max(epochs_all))
            
            iso = argsort(acc_vl_all)[-nof_2nd_tune_models:]
            combinations = [combinations[it] for it in iso]
            avail_thr = os.cpu_count()-threading.active_count()
            bl_thr = ones(nof_2nd_tune_models, dtype=int)
            while sum(bl_thr)<avail_thr:
                for ithr in range(len(bl_thr)):
                    if sum(bl_thr)<avail_thr:
                        bl_thr[ithr] += 1
            print("bl_thr",bl_thr,flush=True)
            acc_tr_all, acc_vl_all, acc_te_all, epochs_all = run_mult_proc_dann(combinations,
                    nof_folds,Xtr,ytr,Xte,yte,tr_inds,vl_inds,path_,nof_2nd_tune_epochs,bl_thr,IS_HPC)
            print("2 multiprocessing done",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3], flush=True)
            for istrcomb in range(len(combinations[0])):
                comb_i = [combo[istrcomb] for combo in combinations]
                print(str_combs[istrcomb],"min=", min(comb_i),"mean=", mean(comb_i),"max=", max(comb_i))
            print("epochs_all","min=", min(epochs_all),"mean=", mean(epochs_all),"max=", max(epochs_all))

        print("***********************************************",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3],
              "***********************************************",flush=True)
        imax = argmax(acc_vl_all)
        best_combination = (combinations[imax][0],combinations[imax][1],int(epochs_all[imax]),
                            combinations[imax][2],combinations[imax][3],combinations[imax][4],combinations[imax][5])
        print("layers, neurons, epochs, learning_rate, dropout, batch, moment_um", flush=True)
        print(imax,"Best Combination:",best_combination, flush=True)

        ttr = time()-t0
        # Save the scaler to a file
        with open(path_ + "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        if IS_HPC:
            with threadpool_limits(limits=nof_final_blas_thr, user_api='blas'):
                history, model = train_pytorch(Xtr, ytr, Xtr, ytr, Xte, yte, combinations[imax], int(epochs_all[imax]), path_, True)
        else:
            history, model = train_pytorch(Xtr, ytr, Xtr, ytr, Xte, yte, combinations[imax], int(epochs_all[imax]), path_, True)
        print("Final Train done",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3], flush=True)
        
        plot_history(history,100,'Train','Test',path_)
        plot_history(history,50,'Train','Test',path_)
        
        pred_tr = pred_DANN(Xtr)
        t0=time()
        for i in range(10):
            pred_te = pred_DANN(Xte)
        tte=(time()-t0)/10
        
        plot_cv_history_dann(acc_tr_all, acc_vl_all, acc_te_all, path_, "Final")
        
        do_sensitivity(Xtr, features_names, target_name, pred_DANN, __method__, path_sens)

        plot_mae_per_bin(ytr, yte, pred_te, target_name, __method__, path_)

        error_analysis(ytr,pred_tr,target_name,__method__,"Train",path_err)
        error_analysis(yte,pred_te,target_name,__method__,"Test",path_err)    

        plot_target_vs_predicted(ytr, pred_tr, target_name, __method__, "Train",path_)  
        plot_target_vs_predicted(yte, pred_te, target_name, __method__, "Test",path_) 
        export_metrics(ytr, pred_tr, yte, pred_te, __method__, ttr, tte, LOGISTIC_REGR, path_)  
        
        print("See results in folder:", path_, flush=True)
        
        # export_notebook_to_html()
        gather_all_ML_metrics(ROOT_DIR)
    except Exception as ex1:
        print(ex1)


def predict_DANN(Xout, yout, target_name, LOGISTIC_REGR, ROOT_DIR):
    global model
    __method__ = "DANN"
    path_ = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep
    path_pred = ROOT_DIR+"Predict"+os.sep+__method__+os.sep
    # try to load the results from the CSV file
    try:
        # Load the scaler from the file
        with open(path_ + "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        Xout = scaler.transform(Xout)

        # load model
        model = torch.load(path_ + "model.pth")

        pred_out = pred_DANN(Xout)
        
    except Exception as e:
        print("Error: ", e)
        return

    # save predictions to file
    with open(path_pred + "Predictions_"+__method__+".csv", "w") as file:
        for yi in pred_out:
            file.write(str(yi) + '\n')

    plot_target_vs_predicted(yout, pred_out, target_name, __method__, "Out", path_pred) 
    export_metrics_out(yout, pred_out, path_pred + __method__ + "_Out", LOGISTIC_REGR)
    error_analysis(yout, pred_out, target_name, __method__, "Out", path_pred)    
    
    print("See results in folder: ", path_pred, flush=True)