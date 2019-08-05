from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
# import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

TRAINING_DAYS = 1000


class BaseModel:

    def __init__(self):
        self._register_classifier()

    def run(self, ohlc_df):
        # remove most recent day and learn on rest
        last_day = ohlc_df.iloc[-1]
        learning_df = ohlc_df.iloc[:-1]
        self.learn(learning_df)
        return self.predict(last_day)

    def _register_classifier(self):
        raise NotImplemented

    def learn(self, df):
        '''
        fits the classifier by providing input data and classification

        Additional input features could be added to the input dataframe
        here
        '''

        def make_signed_int(x):
            try:
                return int(x / abs(x))

            except ZeroDivisionError:
                return 0

        # first day discarded since has no 'daily_change'
        df = df.iloc[-TRAINING_DAYS:]
        # first day discarded since has no 'daily_change'
        daily_changes = (df.close - df.close.shift(1)).iloc[1:]
        daily_changes = daily_changes.dropna()
        df = df.iloc[1:]
        up_or_down = daily_changes.apply(make_signed_int)
        self._classifier.fit(df, up_or_down)

    def predict(self, latest_data):
        # wrap in list to make it 2D data which sklearn expects
        return self._classifier.predict([latest_data])[0]


class EnsembleModel(BaseModel):

    def _register_classifier(self):
        # estimators = [
        #    ('logistic', LogisticRegression()),
        #    ('gradientboosting', GradientBoostingClassifier()),
        # ]
        # self._classifier = VotingClassifier(estimators)

        logisticregression = LogisticRegression()
        ada = AdaBoostClassifier(base_estimator=logisticregression, n_estimators=50, learning_rate=0.5, random_state=42)
        self._classifier = BaggingClassifier(base_estimator=ada)

        # Four boosting/ensemble models to test:

        # 1. self._classifier = xgb.XGBClassifier(max_depth=5, n_estimators=10000, learning_rate=0.3, n_jobs=-1)

        # 2. self._classifier = GradientBoostingClassifier()

        # 3. ADABOOST WITH RANDOM FOREST ENSEMBLE MODEL
        # self._classifier = AdaBoostClassifier(RandomForestClassifier())

        # 4. ADABOOST WITH RANDOM FOREST & BAGGING (Ensemble Model 2)
        # forest = RandomForestClassifier()
        # ada = AdaBoostClassifier(base_estimator=forest, n_estimators=100,learning_rate=0.5, random_state=42)
        # self._classifier = BaggingClassifier()
        # or try to add with parameters, if possible:
        # self._classifier = BaggingClassifier(base_estimator=ada, n_estimators=50,max_samples=1.0, max_features=1.0, bootstrap=True,bootstrap_features=False, n_jobs=-1, random_state=42)


