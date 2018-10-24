The program takes two .csv files, containing sound recordings, as inputs. Principal 
Component Analysis was used to reduce the inputs to a single .csv file output. The algorithm 
builds a covariance-variance matrix based on the inputs, and then determines the largest 
eigen-vector of that matrix. The corresponding eigen-vector, v, as the weights for the 
output. The output was determined by y = x1*v1 + x2v2, where x1 and x2 correspond to a set
of inputs. 

Final weight vector = (1 0.659989734)