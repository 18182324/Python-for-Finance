companies_dict = {
    'Amazon':'AMZN',
    'Apple':'AAPL',
    'Walgreen':'WBA',
    'Northrop Grumman':'NOC',
    'Boeing':'BA',
    'Lockheed Martin':'LMT',
    'McDonalds':'MCD',
    'Intel':'INTC',
    'Navistar':'NAV',
    'IBM':'IBM',
    'Texas Instruments':'TXN',
    'MasterCard':'MA',
    'Microsoft':'MSFT',
    'General Electrics':'GE',
    'Symantec':'SYMC',
    'American Express':'AXP',
    'Pepsi':'PEP',
    'Coca Cola':'KO',
    'Johnson & Johnson':'JNJ',
    'Toyota':'TM',
    'Honda':'HMC',
    'Mistubishi':'MSBHY',
    'Sony':'SNE',
    'Exxon':'XOM',
    'Chevron':'CVX',
    'Valero Energy':'VLO',
    'Ford':'F',
    'Bank of America':'BAC'}

data_source = ‘yahoo’ # Source of data is yahoo finance.
start_date = ‘2015–01–01’ 
end_date = ‘2017–12–31’
df = data.DataReader(list(companies_dict.values()),
 data_source,start_date,end_date)

stock_open = np.array(df[‘Open’]).T # stock_open is numpy array of transpose of df['Open']
stock_close = np.array(df[‘Close’]).T # stock_close is numpy array of transpose of df['Close']
stock_open = np.array(df[‘Open’]).T # stock_open is numpy array of transpose of df['Open']
stock_close = np.array(df[‘Close’]).T # stock_close is numpy array of transpose of df['Close']
sum_of_movement = np.sum(movements,1)

#Print Company and its ‘sum_of_movement’
for i in range(len(companies)):
 print(‘company:{}, Change:{}’.format(df[‘High’].columns[i],sum_of_movement[i]))

#Visualizing Data
plt.figure(figsize = (20,10)) 
plt.subplot(1,2,1) 
plt.title(‘Company:Amazon’,fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)
plt.xlabel(‘Date’,fontsize = 15)
plt.ylabel(‘Opening price’,fontsize = 15)
plt.plot(df[‘Open’][‘AMZN’])
plt.subplot(1,2,2) 
plt.title(‘Company:Apple’,fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)
plt.xlabel(‘Date’,fontsize = 15)
plt.ylabel(‘Opening price’,fontsize = 15)
plt.plot(df[‘Open’][‘AAPL’])

#Plot AMZN
plt.figure(figsize = (20,10)) # Adjusting figure size
plt.title(‘Company:Amazon’,fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)
plt.xlabel(‘Date’,fontsize = 20)
plt.ylabel(‘Price’,fontsize = 20)
plt.plot(df.iloc[0:30][‘Open’][‘AMZN’],label = ‘Open’) # Opening prices of first 30 days are plotted against date
plt.plot(df.iloc[0:30][‘Close’][‘AMZN’],label = ‘Close’) # Closing prices of first 30 days are plotted against date

plt.figure(figsize = (20,8)) 
plt.title('Company:Amazon',fontsize = 20)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 20)
plt.xlabel('Date',fontsize = 20)
plt.ylabel('Movement',fontsize = 20)
plt.plot(movements[0][0:30])

plt.figure(figsize = (20,10)) 
plt.title(‘Company:Amazon’,fontsize = 20)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 20)
plt.xlabel(‘Date’,fontsize = 20)
plt.ylabel(‘Volume’,fontsize = 20)
plt.plot(df[‘Volume’][‘AMZN’],label = ‘Open’)

#Candlestick Chart for First 60 Days
fig = go.Figure(data=[go.Candlestick(x=df.index,
 open=df.iloc[0:60][‘Open’][‘AMZN’],
 high=df.iloc[0:60][‘High’][‘AMZN’],
 low=df.iloc[0:60][‘Low’][‘AMZN’],
 close=df.iloc[0:60][‘Close’][‘AMZN’])])

#Plot Variation of Movement for Amazon and Apple
plt.figure(figsize = (20,8)) 
ax1 = plt.subplot(1,2,1)
plt.title(‘Company:Amazon’,fontsize = 20)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 20)
plt.xlabel(‘Date’,fontsize = 20)
plt.ylabel(‘Movement’,fontsize = 20)
plt.plot(movements[0]) 
plt.subplot(1,2,2,sharey = ax1)
plt.title(‘Company:Apple’,fontsize = 20)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 20)
plt.xlabel(‘Date’,fontsize = 20)
plt.ylabel(‘Movement’,fontsize = 20)

normalizer = Normalizer() # Define a Normalizer
norm_movements = normalizer.fit_transform(movements) # Fit and transform

#Print min,max and mean
print(norm_movements.min())
print(norm_movements.max())

#Create pipeline that normalizes data and applies K-Means clustering algorithm
# Import the necessary packages
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
# Define a normalizer
normalizer = Normalizer()
# Create Kmeans model
kmeans = KMeans(n_clusters = 10,max_iter = 1000)
# Make a pipeline chaining normalizer and kmeans
pipeline = make_pipeline(normalizer,kmeans)
# Fit pipeline to daily stock movements
pipeline.fit(movements)
labels = pipeline.predict(movements)
print(norm_movements.mean())
plt.plot(movements[1])
fig.show()
plt.legend(loc=’upper left’, frameon=False,framealpha=1,prop={‘size’: 22}) # Properties of legend box

#Print company and cluster number
df1 = pd.DataFrame({‘labels’:labels,’companies’:list(companies)}).sort_values(by=[‘labels’],axis = 0)

# Define a normalizer
normalizer = Normalizer()
# Reduce the data

#Plot decision boundary
from sklearn.decomposition import PCA
# Reduce the data
reduced_data = PCA(n_components = 2).fit_transform(norm_movements)
# Define step size of mesh
h = 0.01
# Plot the decision boundary
x_min,x_max = reduced_data[:,0].min()-1, reduced_data[:,0].max() + 1
y_min,y_max = reduced_data[:,1].min()-1, reduced_data[:,1].max() + 1
xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
# Obtain labels for each point in the mesh using our trained model
Z = kmeans.predict(np.c_[xx.ravel(),yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
# Define color plot
cmap = plt.cm.Paired
# Plotting figure
plt.clf()
plt.figure(figsize=(10,10))
plt.imshow(Z,interpolation = ‘nearest’,extent=(xx.min(),xx.max(),yy.min(),yy.max()),cmap = cmap,aspect = ‘auto’,origin = ‘lower’)
plt.plot(reduced_data[:,0],reduced_data[:,1],’k.’,markersize = 5)
# Plot the centroid of each cluster as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0],centroids[:,1],marker = ‘x’,s = 169,linewidths = 3,color = ‘w’,zorder = 10)
plt.title(‘K-Means clustering on stock market movements (PCA-Reduced data)’)
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.show()
reduced_data = PCA(n_components = 2)
# Create Kmeans model
kmeans = KMeans(n_clusters = 10,max_iter = 1000)
# Make a pipeline chaining normalizer, pca and kmeans
pipeline = make_pipeline(normalizer,reduced_data,kmeans)
# Fit pipeline to daily stock movements
pipeline.fit(movements)
# Prediction
labels = pipeline.predict(movements)
# Create dataframe to store companies and predicted labels
df2 = pd.DataFrame({'labels':labels,'companies':list(companies_dict.keys())}).sort_values(by=['labels'],axis = 0)
