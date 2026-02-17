class CorporateExecution:
    def run_workflow(self, data):
        # From docs: ML on FFZ for corporate research
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(data.X, data.y)
        return model.predict(data.test)
