 
from sympy.geometry import ( Line, Point)
 
from sympy import Float 
p1 = Point(1, 1)
 
p3 = Point(2, 2)
p4 = Point(4, 0)
 
 
 
l2_1 = Line(p3, p4)
line3=l2_1.parallel_line(p1) #== Line(Point(0, 0), Point(0, -1))
# print(len(line3))
print(list(line3.args[1]))
# print([i for i in line3])
 