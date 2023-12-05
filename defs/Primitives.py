class Point:
    def __init__(self, x, id, rid, group_id):
        self.x = x
        self.id = id
        self.rid = rid
        self.group_id = group_id

class Map:
    def __init__(self):
        pass
    def shape_function(self, x, y):
        raise NotImplementedError()
    def value(self, x, y):
        return self.shape_function(x,y)
    def gradient_function(self, x, y):
        raise NotImplementedError()
    def gradient(self, x, y):
        return self.gradient_function(self, x, y)
    def area(self):
        raise NotImplementedError()
    def __call__(self, x,y):
        return (self.value(x,y), self.gradient(x,y))

class Integrator:
    def __init__(self):
        pass
    def value_integral(self):
        raise NotImplementedError()
    def gradient_integral(self):
        raise NotImplementedError()

class Element:
    def __init__(self, integrator : Integrator):
        super().__init__()
        self.integrator = integrator


