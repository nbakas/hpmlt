
# Description: This file contains the code for training and evaluating XGBoost models.


from import_libraries import *
from misc_functions import *

def train_xgb_folds(i,max_depth,learning_rate,colsample_bytree,subsample,LOGISTIC_REGR,
               nof_folds,Xtr,ytr,tr_inds,vl_inds,Xte,yte,acc_tr_all,acc_vl_all,acc_te_all,
               nBest_all,nof_tune_rounds,blas_threads,IS_HPC):
    print(i,"started")    
    t0=time()
    if LOGISTIC_REGR:
        xgboost = xgb.XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=nof_tune_rounds, 
                    colsample_bytree=colsample_bytree, subsample=subsample, objective='binary:logistic')
    else: 
        xgboost = xgb.XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, n_estimators=nof_tune_rounds, 
                    colsample_bytree=colsample_bytree, subsample=subsample, objective='reg:squarederror')

    nEstimators = list(concatenate((arange(1,11),arange(20, 101, 10),arange(200, 1001, 100),arange(2000, 10001, 1000))))
    nEstimators = array(nEstimators)
    nEstimators = nEstimators[nEstimators<=nof_tune_rounds]
    nEstimators = nEstimators.tolist()
    if nof_tune_rounds not in nEstimators:
        nEstimators.append(nof_tune_rounds)
    
    acc_tr = zeros(len(nEstimators)); acc_vl = zeros(len(nEstimators)); acc_te = zeros(len(nEstimators))
    for fold in range(nof_folds):
        
        if IS_HPC:
            with threadpool_limits(limits=int(blas_threads[i]), user_api='blas'):
                xgboost.fit(Xtr[tr_inds[fold],:], ytr[tr_inds[fold]])
        else:
            xgboost.fit(Xtr[tr_inds[fold],:], ytr[tr_inds[fold]])
                
        for n,nE in enumerate(nEstimators):
            pred_tr = xgboost.predict(Xtr[tr_inds[fold],:], iteration_range=(0,int(nE)))
            pred_vl = xgboost.predict(Xtr[vl_inds[fold],:], iteration_range=(0,int(nE)))
            pred_te = xgboost.predict(Xte, iteration_range=(0,int(nE)))
            
            # accTr = corrcoef(ytr[tr_inds[fold]], pred_tr)[0,1]
            accTr = 100.0-mean(abs(ytr[tr_inds[fold]] - pred_tr))
            if isnan(accTr):
                accTr = 0
            acc_tr[n] += accTr
            
            # accVl = corrcoef(ytr[vl_inds[fold]], pred_vl)[0,1]
            accVl = 100.0-mean(abs(ytr[vl_inds[fold]] - pred_vl))
            if isnan(accVl):
                accVl = 0
            acc_vl[n] += accVl

            # accTe = corrcoef(yte, pred_te)[0,1]
            accTe = 100.0-mean(abs(yte - pred_te))
            if isnan(accTe):
                accTe = 0
            acc_te[n] += accTe
  
    acc_tr /= nof_folds; acc_vl /= nof_folds; acc_te /= nof_folds
    nBest = argmax(acc_vl)
    acc_tr = acc_tr[nBest]; acc_vl = acc_vl[nBest]; acc_te = acc_te[nBest]; 
    acc_tr_all[i] = acc_tr
    acc_vl_all[i] = acc_vl
    acc_te_all[i] = acc_te
    nBest_all[i] = nEstimators[nBest]
    t1=time()
    sdt = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(sdt,i,"acc_tr",acc_tr,"acc_vl",acc_vl,"acc_te",acc_te,"opti_estimators",nEstimators[nBest],
        "max_depth",max_depth,"learning_rate",learning_rate,"colsample_bytree",colsample_bytree,"subsample",subsample,
        t1-t0,"blas_threads[i]=",blas_threads[i],flush=True)


def run_mult_proc_xgb(combinations,LOGISTIC_REGR,nof_folds,Xtr,ytr,tr_inds,vl_inds,Xte,yte,nof_tune_rounds,blas_threads,IS_HPC):
    manager_tr = multiprocessing.Manager(); acc_tr_all = manager_tr.dict()
    manager_vl = multiprocessing.Manager(); acc_vl_all = manager_vl.dict()
    manager_te = multiprocessing.Manager(); acc_te_all = manager_te.dict()
    manager_nBest = multiprocessing.Manager(); nBest_all = manager_nBest.dict()
    jobs = []
    for i in range(len(combinations)):
        (max_depth, learning_rate, colsample_bytree, subsample) = combinations[i]
        p = multiprocessing.Process(target=train_xgb_folds,
            args=(i,max_depth,learning_rate,colsample_bytree,subsample,LOGISTIC_REGR,
                    nof_folds,Xtr,ytr,tr_inds,vl_inds,Xte,yte,acc_tr_all,acc_vl_all,
                    acc_te_all,nBest_all,nof_tune_rounds,blas_threads,IS_HPC))             
        jobs.append(p)
        p.start()
    
    for proc in jobs:
        proc.join()   
           
    acc_tr_all = array(list(dict(sorted(acc_tr_all.items())).values()))
    acc_vl_all = array(list(dict(sorted(acc_vl_all.items())).values()))
    acc_te_all = array(list(dict(sorted(acc_te_all.items())).values()))
    nBest_all = array(list(dict(sorted(nBest_all.items())).values()))
    return acc_tr_all, acc_vl_all, acc_te_all, nBest_all

def plot_cv_history_xgb(acc_tr_all, acc_vl_all, acc_te_all, path_, tag_):
    iso = argsort(acc_vl_all)
    ipositive = acc_vl_all[iso]>0
    plt.plot(acc_tr_all[iso][ipositive], 'x', label='Train')
    plt.plot(acc_vl_all[iso][ipositive], 'x', label='Validation')
    plt.plot(acc_te_all[iso][ipositive], 'x', label='Test')
    plt.ticklabel_format(style='plain', axis='y')
    plt.legend()
    plt.xlabel('Model')
    plt.ylabel('R2 Score')
    plt.savefig(path_ + "XGBoost_tune_cv_history_" + tag_ + ".png")
    plt.close()
    
    cdf_pdf_plot(acc_vl_all[iso][ipositive],"XGBoost_ValidSet_" + tag_,path_)

    iso = argsort(acc_vl_all)[int(0.5*len(acc_vl_all)):]
    ipositive = acc_vl_all[iso]>0
    plt.plot(acc_tr_all[iso][ipositive], 'x', label='Train')
    plt.plot(acc_vl_all[iso][ipositive], 'x', label='Validation')
    plt.plot(acc_te_all[iso][ipositive], 'x', label='Test')
    plt.ticklabel_format(style='plain', axis='y')
    plt.legend()
    plt.xlabel('Model')
    plt.ylabel('R2 Score')
    plt.savefig(path_ + "XGBoost_tune_cv_history_50perc_" + tag_ + ".png")
    plt.close()
    
    cdf_pdf_plot(acc_vl_all[iso][ipositive],"XGBoost_ValidSet_50perc_" + tag_,path_)


def do_xgboost(Xtr,Xte,ytr,yte,features_names,target_name,IS_HPC,PERMUTE_TRAIN_TEST,LOGISTIC_REGR,ROOT_DIR,
            nof_1st_tune_rounds,nof_1st_tune_models,nof_2nd_tune_rounds,nof_2nd_tune_models,
            nof_final_blas_thr):
    try:
        print("do_xgboost start",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3], flush=True)
        __method__ = "XGBoost"
        path_ = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep
        path_err = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep+"Error_Analysis"+os.sep
        path_sens = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep+"Sensitivity_Analysis"+os.sep
        
        t0=time()
        
        #Tuning Parameters:
        
        # max_depth: It represents the maximum depth of a tree. Increasing this value makes the model 
        # more complex and prone to overfitting. Lower values help prevent overfitting, 
        # but too low values may result in underfitting. 
        # range: [0,$\infty$]. default=6
        max_depth = list(arange(1, 11))
        
        # learning_rate: It controls the step size at each boosting iteration. 
        # A lower learning rate makes the model more robust to overfitting, 
        # but it may require more boosting iterations to converge. 
        # Higher learning rates can lead to faster convergence, but increase the risk of overfitting. 
        # range: [0,1]. default=0.3
        learning_rate = concatenate((   np_round(linspace(0.001,0.01,num=3), 10),
                                        np_round(linspace(0.01,0.1,num=3), 10),
                                        np_round(linspace(0.1,1.0,num=5), 10)))
        learning_rate = list(unique(learning_rate))
        
        # colsample_bytree: It specifies the fraction of columns to be randomly sampled for each tree. 
        # A value less than 1.0 introduces randomness and can help in reducing overfitting. 
        # range: (0, 1]. default=1
        colsample_bytree = list(np_round(arange(0.1, 1.1, 0.1), 10))
        
        # subsample: It represents the fraction of samples (observations) to be randomly selected for each tree. 
        # Lower values make the model more robust to overfitting by introducing randomness. 
        # The default value is 1.0, which means using all samples. 
        # range: (0,1]. default=1
        subsample = list(np_round(arange(0.1, 1.1, 0.1), 10))
        
        combinations = list(product(max_depth, learning_rate, colsample_bytree, subsample)) 
        print("combinations done",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3],"nof combinations=",len(combinations),flush=True)
        # randomly permute the combinations
        if IS_HPC:
            with threadpool_limits(limits=nof_final_blas_thr, user_api='blas'):
                combinations = shuffle(combinations, random_state=0)
        else:
            combinations = shuffle(combinations, random_state=0)
            
        if len(combinations)>nof_1st_tune_models:
            combinations = combinations[:nof_1st_tune_models]
        print("combinations edited",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3],"nof combinations=",len(combinations),flush=True)

        # split train to train and validation
        ___perc_cv___ = 0.8; nof_folds = 5; obs = len(ytr)
        tr_inds, vl_inds = split_tr_vl(obs,___perc_cv___,nof_folds,PERMUTE_TRAIN_TEST)
        print("split_tr_vl done",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3], flush=True)
        
        # train xgboost on each combination and evaluate on validation set
        if not IS_HPC:
            print("I do not run in parallel!",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3], flush=True)
            acc_tr_all = zeros(nof_1st_tune_models)-Inf; acc_vl_all = zeros(nof_1st_tune_models)-Inf; 
            acc_te_all = zeros(nof_1st_tune_models)-Inf; nBest_all = zeros(nof_1st_tune_models)
            bl_thr = 8*ones(nof_1st_tune_models, dtype=int)
            for i, (max_depth, learning_rate, colsample_bytree, subsample) in enumerate(combinations):
                train_xgb_folds(i,max_depth,learning_rate,colsample_bytree,subsample,LOGISTIC_REGR,
                                nof_folds,Xtr,ytr,tr_inds,vl_inds,Xte,yte,acc_tr_all,acc_vl_all,acc_te_all,
                                nBest_all,nof_1st_tune_rounds,bl_thr,IS_HPC)
            print("1st multiprocessing done",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3], flush=True)
            plot_cv_history_xgb(acc_tr_all, acc_vl_all, acc_te_all, path_, "Initial")
            str_combs = "max_depth", "learning_rate", "colsample_bytree", "subsample"
            for istrcomb in range(len(combinations[0])):
                comb_i = [combo[istrcomb] for combo in combinations]
                print(str_combs[istrcomb],"min=", min(comb_i),"mean=", mean(comb_i),"max=", max(comb_i))
            print("opti-epochs","min=", min(nBest_all),"mean=", mean(nBest_all),"max=", max(nBest_all))
            
            iso = argsort(acc_vl_all)[-nof_2nd_tune_models:]
            combinations = [combinations[it] for it in iso]
            acc_tr_all = zeros(nof_2nd_tune_models)-Inf; acc_vl_all = zeros(nof_2nd_tune_models)-Inf; 
            acc_te_all = zeros(nof_2nd_tune_models)-Inf; nBest_all = zeros(nof_2nd_tune_models)
            bl_thr = 8*ones(nof_1st_tune_models, dtype=int)
            for i, (max_depth, learning_rate, colsample_bytree, subsample) in enumerate(combinations):
                train_xgb_folds(i,max_depth,learning_rate,colsample_bytree,subsample,LOGISTIC_REGR,
                                nof_folds,Xtr,ytr,tr_inds,vl_inds,Xte,yte,acc_tr_all,acc_vl_all,acc_te_all,
                                nBest_all,nof_2nd_tune_rounds,bl_thr,IS_HPC)
            print("2nd multiprocessing done",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3], flush=True)
            for istrcomb in range(len(combinations[0])):
                comb_i = [combo[istrcomb] for combo in combinations]
                print(str_combs[istrcomb],"min=", min(comb_i),"mean=", mean(comb_i),"max=", max(comb_i))
            print("opti-epochs","min=", min(nBest_all),"mean=", mean(nBest_all),"max=", max(nBest_all))
            
        else:
            print("I run in parallel!",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3], flush=True)
            avail_thr = os.cpu_count()-threading.active_count()
            bl_thr = ones(nof_1st_tune_models, dtype=int)
            while sum(bl_thr)<avail_thr:
                for ithr in range(len(bl_thr)):
                    if sum(bl_thr)<avail_thr:
                        bl_thr[ithr] += 1
            print("bl_thr",bl_thr,flush=True)
            acc_tr_all, acc_vl_all, acc_te_all, nBest_all = run_mult_proc_xgb(combinations,
                        LOGISTIC_REGR,nof_folds,Xtr,ytr,tr_inds,vl_inds,Xte,yte,nof_1st_tune_rounds,bl_thr,IS_HPC)
            plot_cv_history_xgb(acc_tr_all, acc_vl_all, acc_te_all, path_, "Initial")
            print("1 multiprocessing done",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3], flush=True)
            str_combs = "max_depth", "learning_rate", "colsample_bytree", "subsample"
            for istrcomb in range(len(combinations[0])):
                comb_i = [combo[istrcomb] for combo in combinations]
                print(str_combs[istrcomb],"min=", min(comb_i),"mean=", mean(comb_i),"max=", max(comb_i))
            print("opti-epochs","min=", min(nBest_all),"mean=", mean(nBest_all),"max=", max(nBest_all))
            
            iso = argsort(acc_vl_all)[-nof_2nd_tune_models:]
            combinations = [combinations[it] for it in iso]
            avail_thr = os.cpu_count()-threading.active_count()
            bl_thr = ones(nof_2nd_tune_models, dtype=int)
            while sum(bl_thr)<avail_thr:
                for ithr in range(len(bl_thr)):
                    if sum(bl_thr)<avail_thr:
                        bl_thr[ithr] += 1
            print("bl_thr",bl_thr,flush=True)
            acc_tr_all, acc_vl_all, acc_te_all, nBest_all = run_mult_proc_xgb(combinations,
                        LOGISTIC_REGR,nof_folds,Xtr,ytr,tr_inds,vl_inds,Xte,yte,nof_2nd_tune_rounds,bl_thr,IS_HPC)
            print("2 multiprocessing done",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3], flush=True)
            for istrcomb in range(len(combinations[0])):
                comb_i = [combo[istrcomb] for combo in combinations]
                print(str_combs[istrcomb],"min=", min(comb_i),"mean=", mean(comb_i),"max=", max(comb_i))
            print("opti-epochs","min=", min(nBest_all),"mean=", mean(nBest_all),"max=", max(nBest_all))
            
        imax = argmax(acc_vl_all)
        best_combination = (combinations[imax][0],combinations[imax][1],nBest_all[imax],combinations[imax][2],combinations[imax][3])
        print("max_depth, learning_rate, n_estimators, colsample_bytree, subsample")
        print(imax,"Best Combination:",best_combination)

        with open(path_ + "Best_Combination.txt", "w") as f:
            f.write(str(best_combination) + "\n")
            f.write("max_depth, learning_rate, n_estimators, colsample_bytree, subsample" + "\n")

        (max_depth, learning_rate, n_estimators, colsample_bytree, subsample) = best_combination
        if LOGISTIC_REGR:
            xgboost = xgb.XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, 
                                    colsample_bytree=colsample_bytree, subsample=subsample, objective='binary:logistic')
        else: 
            xgboost = xgb.XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, n_estimators=int(n_estimators), 
                colsample_bytree=colsample_bytree, subsample=subsample, objective='reg:squarederror')
        
        t0p=time()
        if IS_HPC:
            with threadpool_limits(limits=nof_final_blas_thr, user_api='blas'):
                xgboost.fit(Xtr, ytr)
        else:
            xgboost.fit(Xtr, ytr)
        print("Final Train done",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3], flush=True)
        tfinal = time()-t0p
            
        with open(path_ + "best_estimator_xgb.pkl", 'wb') as f:
            pickle.dump(xgboost, f)
        ttr = time()-t0
        print("train done",datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3], flush=True)
        
        pred_tr = xgboost.predict(Xtr)
        t0=time()
        for i in range(10):
            pred_te = xgboost.predict(Xte)
        tte=(time()-t0)/10
        # get the feature importance from best_xgb and plot it
        feature_importance = xgboost.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = argsort(feature_importance)
        pos = arange(sorted_idx.shape[0]) + .5
        plt.figure(figsize=(7, 7), dpi=500)
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, features_names[sorted_idx])
        plt.tight_layout()
        plt.savefig(path_ + "features_importance_xgb.png")
        plt.close()

        plot_cv_history_xgb(acc_tr_all, acc_vl_all, acc_te_all, path_, "Final")

        do_sensitivity(Xtr, features_names, target_name, xgboost.predict, __method__, path_sens)

        plot_mae_per_bin(ytr, yte, pred_te, target_name, __method__, path_)

        error_analysis(ytr,pred_tr,target_name,__method__,"Train",path_err)
        error_analysis(yte,pred_te,target_name,__method__,"Test",path_err)    

        plot_target_vs_predicted(ytr, pred_tr, target_name, __method__, "Train",path_)  
        plot_target_vs_predicted(yte, pred_te, target_name, __method__, "Test",path_) 
        export_metrics(ytr, pred_tr, yte, pred_te, __method__, ttr, tte, LOGISTIC_REGR, path_)  
        
        print("See results in folder:", path_)
        
        gather_all_ML_metrics(ROOT_DIR)
        
    except Exception as ex1:
        print(ex1)

    

def predict_xgboost(Xout, yout, target_name, LOGISTIC_REGR, ROOT_DIR):
    __method__ = "XGBoost"
    path_ = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep
    path_pred = ROOT_DIR+"Predict"+os.sep+__method__+os.sep
    try:
        with open(path_ + "best_estimator_xgb.pkl", 'rb') as f:
            best_estimator_xgb = pickle.load(f)
    except Exception as e:
        print("Error: ", e)
        return

    pred_out = best_estimator_xgb.predict(Xout)

    # save predictions to file
    with open(path_pred + "Predictions_"+__method__+".csv", "w") as file:
        for yi in pred_out:
            file.write(str(yi) + '\n')

    plot_target_vs_predicted(yout, pred_out, target_name, __method__, "Out", path_pred) 
    export_metrics_out(yout, pred_out, path_pred + __method__ + "_Out", LOGISTIC_REGR)
    error_analysis(yout, pred_out, target_name, __method__, "Out", path_pred)    
      
    
    print("See results in folder: ", path_pred)


def do_QuantileGradientBoostingRegressor(ROOT_DIR, Xtr, Xte, ytr, yte, target_name):
    # open the text file
    __method__ = "XGBoost"
    path_ = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep
    with open(path_ + 'Best_Combination.txt') as file:
        values = file.read()
    values = values.split("\n")[0]
    values = tuple(map(float, values.strip('()').split(', ')))
    print("Best_Combination:",values)
    # extract the parameters from the values
    max_depth__ = int(values[0])
    learning_rate__ = float(values[1])
    n_estimators__ = int(values[2])
    colsample_bytree__ = float(values[3])
    subsample__ = float(values[4])

    common_params = dict(
        learning_rate=learning_rate__,
        n_estimators=n_estimators__,
        max_depth=max_depth__,
        subsample = subsample__,
        max_features = colsample_bytree__
    )
    
    alpha_low = 0.05
    print('Training Q'+str(round(100*alpha_low,0))+'%')
    gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha_low, **common_params)
    model_low = gbr.fit(Xtr, ytr)
    
    alpha_med = 0.5
    print('Training Q'+str(round(100*alpha_med,0))+'%')
    gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha_med, **common_params)
    model_med = gbr.fit(Xtr, ytr)
    
    alpha_high = 0.95
    print('Training Q'+str(round(100*alpha_high,0))+'%')
    gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha_high, **common_params)
    model_high = gbr.fit(Xtr, ytr)
    
    plt.figure(figsize=(200, 20))
    plt.plot(yte, label=target_name + ' Test Set', marker='x')
    plt.plot(model_low.predict(Xte), label='Q'+str(round(100*alpha_low,0))+'%', marker='v')
    plt.plot(model_med.predict(Xte), label='Q'+str(round(100*alpha_med,0))+'%', marker='x')
    plt.plot(model_high.predict(Xte), label='Q'+str(round(100*alpha_high,0))+'%', marker='^')
    plt.grid()
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.legend(fontsize=50)
    plt.tight_layout()
    plt.savefig(path_ + "QuantileGradientBoostingRegressor_Test.png")
    plt.close()

    # Save the trained model to a file
    with open(path_ + "model_low.pkl", "wb") as f:
        pickle.dump(model_low, f)
    # Save the trained model to a file
    with open(path_ + "model_med.pkl", "wb") as f:
        pickle.dump(model_med, f)
    # Save the trained model to a file
    with open(path_ + "model_high.pkl", "wb") as f:
        pickle.dump(model_high, f)
        
def predict_quantile_gb(ROOT_DIR, Xout):
    __method__ = "XGBoost"
    path_ = ROOT_DIR+"ML_Models"+os.sep+__method__+os.sep
    path_pred = ROOT_DIR+"Predict"+os.sep+__method__+os.sep
    # Load the trained model_low from the file
    with open(path_ + "model_low.pkl", "rb") as f:
        model_low = pickle.load(f)
    # Use the loaded model to make predictions on new data
    y_pred_low = model_low.predict(Xout)
    
    # Load the trained model_med from the file
    with open(path_ + "model_med.pkl", "rb") as f:
        model_med = pickle.load(f)
    # Use the loaded model to make predictions on new data
    y_pred_med = model_med.predict(Xout)
    
    # Load the trained model_high from the file
    with open(path_ + "model_high.pkl", "rb") as f:
        model_high = pickle.load(f)
    # Use the loaded model to make predictions on new data
    y_pred_high = model_high.predict(Xout)
    
    # Create a DataFrame with the predictions
    df = pd.DataFrame({ 'y_pred_low': y_pred_low,
                        'y_pred_med': y_pred_med,
                        'y_pred_high': y_pred_high})
    # Save the DataFrame to an Excel file
    df.to_excel(path_pred + 'predict_quantile_gb.xlsx', index=False)