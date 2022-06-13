class Company(object):
    def __init__(self):
        super(self, Company).__init__()
        self.id = None
        self.enterprise_opening_date = None
        self.code_enterprise_ratal_classes = None
        self.code_enterprise_registration = None
        self.HY_DM = [0,0,0,0]  # 行业代码、中类、大类、门类
        self.tax_return = []

    def add_tax_return(self, data):
        