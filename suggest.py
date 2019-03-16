import csv
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, cross_validation,neighbors
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer # VADER https://github.com/cjhutto/vaderSentiment


def processDataForCompany(symbol):
    futr_days=5
    df2 = pd.read_csv('sp500_adjcloses.csv',index_col=0)
    symbols = df2.columns.values.tolist()
    
    #sentiment analysis of news and finally classifying this score into -1,0,1 
    df1 = pd.read_csv("Combined_News_DJIA.csv")
    df1.describe()
    df1.Date = pd.to_datetime(df1.Date)
    df1.head()
    df1.index = df1.Date

    scores = pd.DataFrame(index = df1.Date, columns = ['Compound'])
    analyzer = SentimentIntensityAnalyzer() # Use the VADER Sentiment Analyzer

    for j in range(1,df1.shape[0]):
        tmp_comp = 0
        for i in range(2,df1.shape[1]):
            text = df1.iloc[j,i]
            if(str(text) == "nan"):
                tmp_comp +=  0

            else:
                vs = analyzer.polarity_scores(df1.iloc[j,i])
                tmp_comp +=  vs['compound']

        tmp_comp = tmp_comp/25
        scores.iloc[j,] = [tmp_comp]

    scores = scores.dropna()
    df = scores.join(df2)
    df.fillna(-9999,inplace=True)
    for i in range(1,futr_days+1):
        df['{}.{}d'.format(symbol,i)] = (df[symbol].shift(-i) - df[symbol])/df[symbol]
        df['Compound.{}d'.format(i)] = df['Compound'].shift(-i)
    df.fillna(-9999,inplace=True)

    return symbols,df,futr_days


def compare(*args):
    cols = [c for c in args]
    threshold = 0.027
    for actual_change in cols:
        if actual_change < -threshold:
            return -1
        if actual_change > threshold:
            return 1
    return 0


def bollinger_change(sy,lower,upper):
    if sy > upper:
        return -1
    if sy < lower:
        return 1
    return 0

def compare2(comp1,comp2):

    if comp2 > comp1:
    	return 1
    if comp1 > comp2:
    	return -1
    return 0


# processDataforCompany()
#calculating other factors and also classifying their score as -1,0,1
def get_train_data(symbol):
    symbols, df,pred_days = processDataForCompany(symbol)
    for i in range(1,pred_days+1):
        df['{}.pos_neg_zero'.format(symbol)] = list(map(compare,
                                    df['{}.{}d'.format(symbol,i)],
                                                  ))
    
    df['com_pnz'] = list(map(compare2,
                                    df['Compound'],df['Compound.{}d'.format(i)],
                                                  ))

    #BOLLINGER_BANDS
    df['{}_30DayMA'.format(symbol)] = df[symbol].rolling(window=20).mean()
    df['{}_30DaySTD'.format(symbol)] = df[symbol].rolling(window=20).std()
    df['{}_UpperBand'.format(symbol)] = df['{}_30DayMA'.format(symbol)] + (df['{}_30DaySTD'.format(symbol)] * 2)
    df['{}_LowerBand'.format(symbol)] = df['{}_30DayMA'.format(symbol)] - (df['{}_30DaySTD'.format(symbol)] * 2)
    df['boll_pnz'] = list(map(bollinger_change,df[symbol],df['{}_LowerBand'.format(symbol)],df['{}_UpperBand'.format(symbol)]))

    # calculating MACD factor
    df['ema26'] = df[symbol].ewm(span=26).mean()
    df['ema12'] = df[symbol].ewm(span=12).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal_line'] = df['macd'].ewm(span=9).mean()

    df['macd_pnz'] = list(map(compare2,
                                    df['macd'],df['signal_line'],
                                                  ))
    
    vals = df['{}.pos_neg_zero'.format(symbol)].values.tolist()

    df.fillna(-9999,inplace=True)
    df_final1 = df[['{}.pos_neg_zero'.format(symbol),'macd_pnz','boll_pnz','com_pnz']]
    df_final2 = pd.DataFrame(index = df_final1.index, columns = ['final_pnz'])
    df_final = df_final1.join(df_final2)
    length = len(df_final)
    
    #for loop to get the value which occurs maximum
    #time among -1,0,1 in all factor
    for i in range(0,length):
        count_pos = 0
        count_neg = 0
        count_zero = 0
        for j in range(0,4):
            if df_final.iat[i,j] == 1:
                count_pos += 1
            if df_final.iat[i,j] == -1:
        	    count_neg += 1
            if df_final.iat[i,j] == 0:
        	    count_zero += 1
        if count_pos > count_neg and count_pos > count_zero:
            df_final.iat[i,4] = 1
        elif count_neg > count_pos and count_neg > count_zero:
            df_final.iat[i,4] = -1
        else:
            df_final.iat[i,4] = 0

    df_final = df_final.join(df[symbol])
    df_vals = df[[symbol for symbol in symbols]]
    df_vals.fillna(-9999,inplace=True)
    X = df_vals.values
    Y = df_final['final_pnz'].values
    return X,Y,df_final,symbols


#extract_featuresets('GOOGL')
#backtesting the algorithm performance
def backtest(df_final,symbol):
    totalStocks = 0
    # print(df_final.head())
    startingCapital = df_final.iat[1,5] * 8 #first closing price
    funds = startingCapital
    currentValuation = funds
    length = len(df_final)

    perf = []
    date = []
    perc = []
    date.append("date")
    perf.append("currentValuation")
    perc.append("perChange")
    for i in range(0,length):
        index = df_final.index[i]
        try:
            price = df_final[symbol][i]
            change = df_final.iat[i,4]

            if change > 0:
                if (change * price) < funds:
                    funds -= (change * price)
                    totalStocks += change
                    currentValuation = funds + (totalStocks * price)
                else:
                    pass

            elif change < 0:

                change = abs(change)

                if (totalStocks - change) <0:
                    change = totalStocks
                    totalStocks = 0
                    funds += (change * price)
                    currentValuation = funds
                else:
                    totalStocks -= change
                    funds += (change * price)
                    currentValuation = funds + (totalStocks * price)

            percChange = round(((currentValuation-startingCapital)/startingCapital) * 100)
            date.append(index)
            perf.append(currentValuation)
            perc.append(percChange)
        except:
            pass
    columns=zip(date,perf,perc)
    with open('performance1.csv', "w") as fl:
        writer = csv.writer(fl)
        for column in columns:
            writer.writerow(column)
 

#seeing the backtest result using graph
def backtest_result():
    df = pd.read_csv('performance1.csv',index_col='date',parse_dates = True)
    res = df['perChange'].rolling(window=50, min_periods=0).mean()
    res.plot(label='performance')
    plt.legend()
    plt.show()


def mlClassifiers(company):
    x,y, df_final,symbols = get_train_data(company)
    Xtrain,Xtest,Ytrain,Ytest = cross_validation.train_test_split(x,y,test_size=0.2)
    X = Xtrain.astype(int)
    Y = Ytrain.astype(int)
    Xt = Xtest.astype(int)
    Yt = Ytest.astype(int)
  
    clf = VotingClassifier([('lsvc',svm.LinearSVC()),('knn',neighbors.KNeighborsClassifier()),('rf',RandomForestClassifier())])
    clf.fit(X,Y)
    accuracy = clf.score(Xt,Yt)
    print('Accuracy', accuracy)
    prediction = clf.predict(Xt)
    print('prediction spread:',Counter(prediction))
    backtest(df_final,company)
    backtest_result()
    return accuracy

# mlClassifiers('GOOGL')
def main():
    company_symb = input("enter the symbol of the company: ")
    mlClassifiers(company_symb)

if __name__ == '__main__':
    main()
