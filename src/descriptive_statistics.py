
from import_libraries import *
from misc_functions import *


def descriptive_statistics(Xtr, ytr, Xte, yte, features_names, target_name, ROOT_DIR):
    try:
        mean1 = mean(c_[Xtr,ytr], axis=0)
        median1 = median(c_[Xtr,ytr], axis=0)
        std1 = std(c_[Xtr,ytr], axis=0)
        min1 = c_[Xtr,ytr].min(axis=0)
        max1 = c_[Xtr,ytr].max(axis=0)
        skewness1 = skew(c_[Xtr,ytr], axis=0)
        kurtosis1 = kurtosis(c_[Xtr,ytr], axis=0)
        
        # test set descriptive statistics
        mean2 = mean(c_[Xte,yte], axis=0)
        median2 = median(c_[Xte,yte], axis=0)
        std2 = std(c_[Xte,yte], axis=0)
        min2 = c_[Xte,yte].min(axis=0)
        max2 = c_[Xte,yte].max(axis=0)
        skewness2 = skew(c_[Xte,yte], axis=0)
        kurtosis2 = kurtosis(c_[Xte,yte], axis=0)
        
        # train, test and differences in one exls file with 3 sheets
        all_names = features_names.append(pd.Index([target_name]))
        file_exp = ROOT_DIR + "Descriptive_Statistics" + os.path.sep + "descriptive_statistics_train_test_and_differences.xlsx"
        writer = pd.ExcelWriter(file_exp, engine='openpyxl')
        
        df1 = pd.DataFrame({'mean':mean1, 'median':median1, 'std':std1, 'min':min1, 'max':max1, 'skewness':skewness1, 
                           'kurtosis':kurtosis1}, index=all_names)
        df1.to_excel(writer, sheet_name='train')
        
        df2 = pd.DataFrame({'mean':mean2, 'median':median2, 'std':std2, 'min':min2, 'max':max2, 'skewness':skewness2, 
                           'kurtosis':kurtosis2}, index=all_names)
        df2.to_excel(writer, sheet_name='test')
        
        df3 = pd.DataFrame({'mean':100*(mean1-mean2)/mean1, 'median':100*(median1-median2)/median1, 'std':100*(std1-std2)/std1, 
                           'min':100*(min1-min2)/min1, 'max':100*(max1-max2)/max1, 'skewness':100*(skewness1-skewness2)/skewness1, 
                           'kurtosis':100*(kurtosis1-kurtosis2)/kurtosis1}, index=all_names)
        df3.to_excel(writer, sheet_name='%differences')
        
        writer.book.save(file_exp)
        writer.close()
        print("Descriptive statistics saved in Descriptive_Statistics/descriptive_statistics_train_test_and_differences.xlsx")

        # convert the dataframe to a LaTeX table
        df_sorted = df1.sort_values(by='kurtosis', key=lambda x: abs(x), ascending=False)
        column_format = '|l|' + '|'.join(['c'] * 7) +'|'
        latex_table = df_sorted.to_latex(column_format=column_format, float_format='%.3g')
        latex_table = latex_table.replace('toprule','hline').replace('bottomrule','hline').replace('midrule','hline\n\\hline')
        latex_table = latex_table.replace('{tabular}','{longtable}')
        with open(ROOT_DIR + "Descriptive_Statistics"+ os.path.sep + "descriptive_statistics_train.txt", "w") as f:
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\caption{Statistical Properties of Variables Features and Target in the Train Set}\n")
            f.write("\\label{tab:descriptive_statistics_train}\n")                                
            f.write(latex_table)
            f.write("\\end{table}\n")
    except Exception as ex1:
        print(ex1)


def plot_pdf_cdf_all(Xtr, ytr, features_names, target_name, ROOT_DIR, set_type):
    try:
        for i in range(Xtr.shape[1]):
            cdf_pdf_plot(Xtr[:,i],features_names[i]+set_type,ROOT_DIR+"Descriptive_Statistics"+os.path.sep+"PDF_CDF"+os.path.sep)
        cdf_pdf_plot(ytr,target_name+set_type,ROOT_DIR+"Descriptive_Statistics"+os.path.sep+"PDF_CDF"+os.path.sep)
        print("PDF and CDF saved in Descriptive_Statistics/PDF_CDF")
    except Exception as ex1:
        print(ex1)

def plot_all_timeseries(XX, features_names, YY, target_name, plot_type, ROOT_DIR):
    try:
        ############################
        window_size = 30
        ############################
        for i in range(XX.shape[1]):
            rolling_correlation = zeros(len(YY))
            for j in range(window_size,len(YY)):
                rolling_correlation[j] = corrcoef(YY[j-window_size:j],XX[j-window_size:j,i])[0,1]
            rolling_correlation[isnan(rolling_correlation)] = 0

            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(50, 12))
            ax1.plot(rolling_correlation)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Rolling Correlation')
            ax1.grid(True)
            ax1.xaxis.set_major_locator(plt.MultipleLocator(100))
            ax2.plot(XX[:,i], color='blue')
            ax2.set_xlabel('Time')
            ax2.set_ylabel(features_names[i], color='blue')
            ax2.grid(True, which='both', axis='both')
            ax2.xaxis.set_major_locator(plt.MultipleLocator(100))
            ax3 = ax2.twinx()
            ax3.plot(YY, color='red')
            ax3.set_ylabel(target_name, color='red')
            fig.savefig(ROOT_DIR + "Descriptive_Statistics" + os.sep + "TimeSeries" + os.sep + plot_type + "_" + features_names[i] + ".png", bbox_inches='tight')
            plt.close()
            
            moving_average = []
            hor = 365
            for ii in range(len(XX[:,i]), hor, -1):
                moving_average.append(mean(XX[:,i][ii:ii-hor:-1]))
            plt.figure(figsize=(50, 12))
            plt.plot(XX[:,i][:len(XX[:,i])-hor])
            plt.title(plot_type + "_" + features_names[i])
            plt.plot(moving_average)
            plt.title(plot_type + "_" + features_names[i]+"_moving_average"+str(hor))
            plt.savefig(ROOT_DIR + "Descriptive_Statistics" + os.sep + "TimeSeries" + os.sep + plot_type + "_" + features_names[i]+"_moving_average"+str(hor)+".png", bbox_inches='tight')
            plt.close()
            
            print("Ploting time series for " + features_names[i])
            
        moving_average = []
        hor = 365
        for i in range(len(YY), hor, -1):
            moving_average.append(mean(YY[i:i-hor:-1]))
        plt.figure(figsize=(50, 12))
        plt.plot(YY[:len(YY)-hor])
        plt.title(plot_type + "_" + target_name)
        plt.plot(moving_average)
        plt.title(plot_type + "_" + target_name+"_moving_average"+str(hor))
        plt.savefig(ROOT_DIR + "Descriptive_Statistics" + os.sep + "TimeSeries" + os.sep + plot_type + "_" + target_name+"_moving_average"+str(hor)+".png", bbox_inches='tight')
        plt.close()
        
        print("Time series saved in Descriptive_Statistics/TimeSeries")
    except Exception as ex1:
        print(ex1)

def plot_all_by_all_correlation_matrix(Xtr, ytr, features_names, target_name, ROOT_DIR):
    try:
        corr = corrcoef(Xtr, ytr, rowvar=False)
        names_all = features_names.copy().tolist()
        names_all.append(target_name)
        sizeFig = 5 + int(Xtr.shape[1]/4)
        fig = plt.figure(figsize=(sizeFig, sizeFig),dpi=1000)
        ax = fig.add_subplot(111)
        cax = ax.matshow(corr, vmin=-1, vmax=1, aspect='auto', cmap=mpl.colormaps['bwr'])
        fig.colorbar(cax)
        ticks = arange(0,len(names_all),1)
        ax.set_xticks(ticks)
        plt.xticks(rotation=90)
        ax.set_yticks(ticks)
        ax.set_xticklabels(names_all)
        ax.set_yticklabels(names_all)
        ax.grid()
        fig.tight_layout()
        plt.savefig(ROOT_DIR + "Descriptive_Statistics"+ os.path.sep + "All_by_All_Correlation_Matrix.png", bbox_inches='tight')
        plt.close()
        # df = pd.DataFrame(c_[Xtr, ytr], columns = features_names.union([target_name]))
        # sns.pairplot(df, diag_kind='kde')
        # df=0
        # plt.savefig(ROOT_DIR + "Descriptive_Statistics"+ os.path.sep + "All_by_All_Correlation_Matrix_Full.png", bbox_inches='tight')
        # plt.close()
        print("All by All Correlation Matrix saved in Descriptive_Statistics/All_by_All_Correlation_Matrix.png")


        cuthmckee = corr.copy()
        for i in range(cuthmckee.shape[0]):
            cuthmckee[i,i] = 0
        cuthmckee[abs(cuthmckee)<percentile(abs(cuthmckee),75)] = 0
        graph = csr_matrix(cuthmckee)
        aux2 = reverse_cuthill_mckee(graph)
        for i in range(len(aux2)):
            graph[:,i] = graph[aux2,i]
        for i in range(len(aux2)):
            graph[i,:] = graph[i,aux2]
        cuthmckee = graph.todense()

        names_all_cut = array(names_all)[aux2].tolist()
        sizeFig = 5 + int(Xtr.shape[1]/4)
        fig = plt.figure(figsize=(sizeFig, sizeFig), dpi=1000)
        ax = fig.add_subplot(111)
        absMax = abs(cuthmckee).max()
        cax = ax.matshow(cuthmckee, vmin=-absMax, vmax=absMax, aspect='auto', cmap=mpl.colormaps['bwr'])
        fig.colorbar(cax)
        ticks = arange(0,len(names_all_cut),1)
        ax.set_xticks(ticks)
        plt.xticks(rotation=90)
        ax.set_yticks(ticks)
        ax.set_xticklabels(names_all_cut)
        ax.set_yticklabels(names_all_cut)
        ax.grid()
        fig.tight_layout()
        plt.savefig(ROOT_DIR + "Descriptive_Statistics"+ os.path.sep + "All_by_All_Correlation_Cuthill_McKee.png", bbox_inches='tight')
        plt.close()


        corr2 = abs(corr)
        for i in range(corr2.shape[0]):
            for j in range(i+1):
                corr2[i,j] = 0
        top_indices = dstack(unravel_index(argsort(corr2.ravel()), corr2.shape))[0][::-1][:20]
        # write the pairs and their correlations to a file
        with open(ROOT_DIR + "Descriptive_Statistics"+ os.path.sep + "All_by_All_Correlation_Cuthill_McKee.txt", "w") as f:
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\caption{Highet Pearson Correlations among all pairs of Features and Target}\n")
            f.write("\\label{tab:All_by_All_Correlation_Cuthill_McKee}\n")
            f.write("\\begin{tabular}{|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("Pearson & Variable i & Variable j \\\\\n")
            f.write("\\hline\n")
            f.write("\\hline\n")
            # write the pairs and their correlations to the file
            for index in top_indices:
                i = index[0]
                j = index[1]
                name_i = names_all[i].replace("_","\\_").replace("#","\\#")
                name_j = names_all[j].replace("_","\\_").replace("#","\\#")
                if i==corr.shape[0]-1:
                    name_i = "\\textbf{" + name_i + "}"
                if j==corr.shape[0]-1:
                    name_j = "\\textbf{" + name_j + "}"
                f.write(f"${corr[i,j]:.5f}$ & {name_i} & {name_j} \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        
        # scatter plot of all features vs target in one plot
        # fig, ax = plt.subplots(Xtr.shape[1], 1, figsize=(7, 5*Xtr.shape[1]),dpi=500)
        # iso = argsort(abs(corr[:-1,-1]))
        # for k,i in enumerate(iso):
        #     ax[k].scatter(Xtr[:,i], ytr)
        #     ax[k].set_xlabel(features_names[i])
        #     ax[k].set_ylabel(target_name)
        #     ax[k].set_title("Pearson Correlation: "+str(round(corr[i,-1],3)))
        #     # add trendline
        #     z = polyfit(Xtr[:,i], ytr, 1)
        #     p = poly1d(z)
        #     ax[k].plot(Xtr[:,i],p(Xtr[:,i]),"r--")
        # fig.tight_layout()
        # plt.savefig(ROOT_DIR + "Descriptive_Statistics"+ os.path.sep + "All_Features_vs_Target.png", bbox_inches='tight')
        # plt.close()
        
        
        iso = argsort(abs(corr[:-1,-1]))
        for k,i in enumerate(iso):
            fig = plt.figure(figsize=(7, 7), dpi=500)
            ax = fig.add_subplot(111)
            ax.scatter(Xtr[:,i], ytr)
            ax.set_xlabel(features_names[i])
            ax.set_ylabel(target_name)
            pearson__ = str(int(100*corr[i,-1]))
            if 0<corr[i,-1]<0.1:
                pearson__ = "0" + pearson__
            if 0>corr[i,-1]>-0.1:
                pearson__ = "-0" + str(int(100*(abs(corr[i,-1]))))
            ax.set_title("Pearson Correlation: "+pearson__+"%")
            # add trendline
            z = polyfit(Xtr[:,i], ytr, 1)
            p = poly1d(z)
            ax.plot(Xtr[:,i],p(Xtr[:,i]),"r--")
            fig.tight_layout()
            plt.savefig(ROOT_DIR+"Descriptive_Statistics"+os.path.sep+"All_Features_vs_Target"+os.path.sep+pearson__+"_"+features_names[i]+".png",
                        bbox_inches='tight')
            plt.close()
        print("All Features vs Target saved in Descriptive_Statistics/All_Features_vs_Target.png")
        
        
        cor_all = corr[:-1,-1]
        iso = argsort(abs(cor_all))
        scf = int(len(features_names)/10)+5
        plt.figure(figsize=(5*scf, 10))
        plt.rcParams.update({'font.size': 5*scf})
        tik = arange(len(cor_all))/7
        plt.bar(tik, cor_all[iso], width=0.1)
        plt.xticks(tik, features_names[iso], rotation=45, ha='right')
        plt.ylabel("Pearson with "+target_name)
        plt.grid()
        plt.tight_layout()
        plt.savefig(ROOT_DIR+"Descriptive_Statistics"+os.path.sep+"All_Features_vs_Target"+os.path.sep+"ALL_CORS_TARGET.png",
                        bbox_inches='tight')
        plt.close()
        plt.rcParams.update({'font.size': 11})
            
    except Exception as ex1:
        print(ex1)

def generate_correlations_map(Xtr,ytr,features_names,target_name,ROOT_DIR):

    names_all = features_names.copy().tolist()
    names_all.append(target_name)

    corr = corrcoef(Xtr, ytr, rowvar=False)
    
    nof_obj = len(names_all)
    similarity = copy(corr)
    for i in range(nof_obj):
        for j in range(nof_obj): 
            similarity[i,j] /= corr[i,i]+corr[j,j]-similarity[i,j]

    func_evals = 2*nof_obj*500
    lb=-10.0*ones((nof_obj,2))
    ub=copy(-lb)
    xa=lb+(ub-lb)*rand(nof_obj,2)
    opti_xa=copy(xa)
    iter_opti = []; all_opti = []
    opti_fu = -Inf
    inz = logical_and((0.0<similarity), (similarity<1))
    opti_D = 0
    for iter in range(func_evals):
        i=randint(0,high=nof_obj)
        j=randint(0,2)
        xa[i,j]=lb[i,j] + rand()*(ub[i,j]-lb[i,j])
        G = xa@transpose(xa)
        g = diag(G).reshape(-1,1)
        D = sqrt(-2*G+g+transpose(g))
        D /= D.max()
        
        fu = corrcoef(D[inz],-log(similarity[inz]))[0,1]
        if fu>opti_fu:
            opti_xa=copy(xa)
            opti_fu=copy(fu)
            opti_D = copy(D)
            iter_opti.append(iter)
            all_opti.append(opti_fu)
        else:
            xa=copy(opti_xa)
        if iter==func_evals-1:
            print(iter, "Map is ready, Optimal Objective: ", opti_fu)

    plt.scatter(iter_opti, all_opti)
    plt.savefig(ROOT_DIR + "Descriptive_Statistics"+ os.path.sep + "Convergence History.png")  
    plt.close()
    plt.scatter(opti_D[inz],-log(similarity[inz]))
    plt.savefig(ROOT_DIR + "Descriptive_Statistics"+ os.path.sep + "Distances vs Similarity.png")  
    plt.close()

    corry = corr[:,-1]
    ss = [int(1000*abs(corry[i])) for i in range(len(corry))]
    fig, ax = plt.subplots(dpi=500)
    plt.scatter(opti_xa[:,0],opti_xa[:,1],s=ss, color='grey')
    texts = [plt.text(opti_xa[i,0], opti_xa[i,1], names_all[i], 
            fontsize=10) for i in range(nof_obj)]
    plt.axis('off')
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'))
    # , arrowprops=dict(arrowstyle='->', color='red')
    plt.savefig(ROOT_DIR + "Descriptive_Statistics"+ os.path.sep + "map.png")  
    plt.close()

    
def export_descriptive_per_bin(Xtr,Xte,ytr,yte,features_names,target_name,ROOT_DIR):

    file_exp = ROOT_DIR + "Descriptive_Statistics" + os.path.sep + "descriptive_statistics_per_bin.xlsx"
    writer = pd.ExcelWriter(file_exp, engine='openpyxl')
                
    percentiles_ytr = []
    percentiles_yte = []
    qstart = [0,0, 0, 0, 50, 75, 95, 99]
    qend =   [1,5,25,50,100,100,100,100]
    for i in range(len(qstart)):
        percentiles_ytr.append((percentile(ytr,qstart[i]), percentile(ytr,qend[i])))
        percentiles_yte.append((percentile(yte,qstart[i]), percentile(yte,qend[i])))
        

    significant_names = []
    significant_perc = []
    str_min_max = []
    for i in range(len(percentiles_ytr)):
        ii = where((percentiles_ytr[i][0]<=ytr) & (ytr<=percentiles_ytr[i][1]))[0]
        str_= '{:.5e}'.format(percentiles_ytr[i][0])+"<=ytr<="+'{:.5e}'.format(percentiles_ytr[i][1])+"|"+str(len(ii))+"_ytr_values"
        DX = (Xtr[ii,:].max(axis=0) - Xtr[ii,:].min(axis=0))/(Xtr.max(axis=0) - Xtr.min(axis=0))
        iso = argsort(DX)
        str_min_max.append(features_names[iso[0]] + "_in_" + '{:.2e}'.format(Xtr[ii,iso[0]].min(axis=0))+"~"+'{:.2e}'.format(Xtr[ii,iso[0]].max(axis=0)))
        mean1 = mean(Xtr[ii,:], axis=0)[iso]
        median1 = median(Xtr[ii,:], axis=0)[iso]
        std1 = std(Xtr[ii,:], axis=0)[iso]
        min1 = Xtr[ii,:].min(axis=0)[iso]
        max1 = Xtr[ii,:].max(axis=0)[iso]
        skewness1 = skew(Xtr[ii,:], axis=0)[iso]
        kurtosis1 = kurtosis(Xtr[ii,:], axis=0)[iso]
        df1 = pd.DataFrame({'Dataset':"Train", 'mean':mean1, 'median':median1, 'std':std1, 'min':min1, 'max':max1, 'skewness':skewness1, 
                            'kurtosis':kurtosis1}, index=features_names[iso])
        significant_names.append(features_names[iso[0]])
        significant_perc.append(DX[iso[0]])
        
        ii = where((percentiles_yte[i][0]<=yte) & (yte<=percentiles_yte[i][1]))[0]
        str_ += "|"+str(len(ii))+"_yte_values"
        mean2 = mean(Xte[ii,:], axis=0)[iso]
        median2 = median(Xte[ii,:], axis=0)[iso]
        std2 = std(Xte[ii,:], axis=0)[iso]
        min2 = Xte[ii,:].min(axis=0)[iso]
        max2 = Xte[ii,:].max(axis=0)[iso]
        skewness2 = skew(Xte[ii,:], axis=0)[iso]
        kurtosis2 = kurtosis(Xte[ii,:], axis=0)[iso]
        df2 = pd.DataFrame({'Dataset':"Test", 'mean':mean2, 'median':median2, 'std':std2, 'min':min2, 'max':max2, 'skewness':skewness2, 
                            'kurtosis':kurtosis2}, index=features_names[iso])
        new_df = pd.concat([df1, df2])
        new_row = pd.DataFrame({'Dataset':[None], 'mean':[None], 'median':[None], 'std':[None], 'min':[None], 'max':[None], 'skewness':[None], 
                            'kurtosis':[None]}, index=[str_])
        new_df = pd.concat([new_df, new_row])
        new_df.to_excel(writer, sheet_name="q"+str(qstart[i])+"-q"+str(qend[i]))

    writer.book.save(file_exp)
    writer.close()


    plt.barh(range(len(significant_names)), significant_perc, align='center')
    plt.yticks(range(len(significant_names)), str_min_max)
    for i, v in enumerate(significant_perc):
        plt.text(v + 0.01, i + 0.25, target_name+" in Q"+str(qstart[i])+"-Q"+str(qend[i]), color='blue', fontweight='bold')
    plt.xlabel("Percentage of Xmax-Xmin in Quantile")
    plt.tight_layout()
    # Get the current axis object
    ax = plt.gca()
    # Remove the top and right spines of the axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(ROOT_DIR + "Descriptive_Statistics"+ os.path.sep + "Percentage_of_Xmax-Xmin_in_Quantile.png")
    plt.close()

