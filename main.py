import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
import feature_selection
import effective_feature_selection
import matplotlib.pyplot as plt
import time


def data_cleaning(df):
    df['protocol_type'] = df['protocol_type'].astype('category')
    df['service'] = df['service'].astype('category')
    df['flag'] = df['flag'].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    probe = ['ipsweep.', 'nmap.', 'portsweep.', 'satan.', 'saint.', 'mscan.']
    dos = ['back.', 'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.', 'mailbomb.', 'udpstorm.', 'apache2.',
           'processtable.']
    r2l = ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'phf.', 'spy.', 'warezclient.', 'warezmaster.',
           'named.', 'snmpguess.', 'worm.', 'snmpgetattack.', 'xsnoop.', 'xlock.', 'sendmail.',
           'buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.', 'xterm.', 'ps.', 'sqlattack.', 'httptunnel.']

    #df.loc[df['type'] == 'normal.', 'type'] = '1'
    df.loc[df['type'].isin(probe), 'type'] = "probe"
    df.loc[df['type'].isin(dos), 'type'] = "dos"
    df.loc[df['type'].isin(r2l), 'type'] = "r2l"

    return df

def scaling(train,test):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(train)
    X_train = scaler.transform(train)
    X_test = scaler.transform(test)
    return X_train, X_test

def support_vector_machine(x_train,x_test):
    svc = BaggingClassifier(svm.SVC(kernel ="linear")
                            ,n_estimators = 10, max_samples = 0.1, max_features = 1.0, bootstrap = True, bootstrap_features = False, warm_start = True, n_jobs = -1, random_state = 2)
    svc.fit(x_train, y_train)
    per = round(svc.score(x_test, y_test) * 100, 3)
    return per


file = 'kddcup.csv'
transformed_file = 'kddcup_tranformed.csv'
dataframe = pd.read_csv(file)
df = data_cleaning(dataframe)
df.to_csv(transformed_file, index=False)

n = 41
chi, info_gain, gain_ratio, symmetric_uncertainty = feature_selection.all_feature(transformed_file)
ensembled_feature = effective_feature_selection.algo(df, n, chi, info_gain, gain_ratio, symmetric_uncertainty)
print(ensembled_feature)

accuracy = []
sample = list(range(1,42))
X = df.iloc[:,:41]
y = df["type"]
test = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size = test, random_state = 1)
X_train, X_test = scaling(X_train,X_test)

for i in range(1,41):
    start = time.time()
    col = ensembled_feature[:i]
    x_train = X_train[:,col]
    x_test = X_test[:,col]
    per = support_vector_machine(x_train,x_test)
    accuracy.append(per)
    print(per)
    print(time.time() - start)

plt.plot(sample,accuracy)
plt.ylabel('Accuracy in Percentage')
plt.xlabel('No of Features')
plt.show()