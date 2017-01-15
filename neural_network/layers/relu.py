class ReLuLayer(object):
    # TODO: gaus distribution
    ''' ReLu: for some nonlinearity. o=max(0,i)'''

    def __init__(self, iid=-1):
        self.iid = iid

    def pprint(self):
        print("ReLu Layer with ID=", self.iid)

    def forward_prop(self, data, debug=False):
        self.i = (data >= 0) * data  # for back_prop
        return (data >= 0) * data

    def back_prop(self, data, debug=False):
        return (self.i >= 0) * data
