        data = pd.read_excel('iris.xlsx')
        df_norm = data[data.columns[1:-1]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        unique = np.unique(data.iloc[:,-1])
        
        target = data.iloc[:,-1].replace(unique,range(len(unique)))
        df = pd.concat([df_norm, target], axis=1)
        train_test_per = (100-size)/100.0
        df['train'] = np.random.rand(len(df)) < train_test_per
        train = df[df.train == 1]
        train = train.drop('train', axis=1).sample(frac=1)
        test = df[df.train == 0]
        test = test.drop('train', axis=1)
        X = train.values[:,:4]
        targets = [[1,0,0],[0,1,0],[0,0,1]]
        y = np.array([targets[int(x)] for x in train.values[:,-1]])

        num_inputs = len(X[0])
        hidden_layer_neurons = 5
        np.random.seed(4)
        w1 = 2*np.random.random((num_inputs, hidden_layer_neurons)) - 1

        num_outputs = len(y[0])
        w2 = 2*np.random.random((hidden_layer_neurons, num_outputs)) - 1

        # TRAINING
        learning_rate = 0.2 
        error = []
        for epoch in range(1000):
            l1 = 1/(1 + np.exp(-(np.dot(X, w1))))
            l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))
            er = (abs(y - l2)).mean()
            error.append(er)
            
            # BACKPROPAGATION / learning!
            # find contribution of error on each weight on the second layer
            l2_delta = (y - l2)*(l2 * (1-l2))
            w2 += l1.T.dot(l2_delta) * learning_rate
            
            l1_delta = l2_delta.dot(w2.T) * (l1 * (1-l1))
            w1 += X.T.dot(l1_delta) * learning_rate
        
        #TEST
        X = test.values[:,:4]
        y = np.array([targets[int(x)] for x in test.values[:,-1]])

        l1 = 1/(1 + np.exp(-(np.dot(X, w1))))
        l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))

        np.round(l2,3)

        # print(w1)
        # print(w2)

        y_pred = np.argmax(l2, axis=1) 
        res = y_pred == np.argmax(y, axis=1)
        correct = np.sum(res)/len(res)

        test_df = test
        test_df[['Species']] = test[['Species']].replace(range(len(unique)), unique)

        test_df['Prediction'] = y_pred
        test_df['Prediction'] = test_df['Prediction'].replace(range(len(unique)), unique)

        acc = correct
        
        ac_label = tk.Label(window2,width=100, text='Accuracy : '+ str(acc),font=my_font1)
        ac_label.grid(row=2,column=0)

        tv2 = ttk.Treeview(window2,height=10)
        tv2.grid(column=0,row=5,padx=5,pady=10,columnspan=10)
        tv2["column"] = list(test_df.columns)
        tv2["show"] = "headings"
        for column in tv2["columns"]:
                tv2.heading(column, text=column)

        df_rows = test_df.to_numpy().tolist()
        for row in df_rows:
            tv2.insert("", "end", values=row)

        cfm = confusion_matrix(test_df[['Species']], test_df[['Prediction']])
        cm = tk.Text(window2)
        cm.delete("1.0","end")
        cm.grid(row=6,column=0)
        cm.insert(tk.END,'Confusion Matrix: \n' + str(cfm)+'\n')
        cm.insert(tk.END,'\na) Recognition rate: '+str(acc*100))
        cm.insert(tk.END,'\nb) Misclassification rate'+str((1-acc)*100)+"\n\n")
        cm.insert(tk.END,classification_report(test_df[['Species']], test_df[['Prediction']]))

        plt.title('Error Graph')
        plt.plot(error)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.show()
        
