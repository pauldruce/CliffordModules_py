# # Clifford Alegrba Generators

# This code create the matrix representations of Clifford algebras.
# The aim of this code to provide everything you need for a Clifford module, given just its type. 
# 
# So far, only the simple cases are coded in, with the procedure to generate arbitrary Clifford algebras generators to be calculated. 
# 
# Note that Python (cmath) uses the letter "j" for the imaginary unit. So we have 
# that a complex number z = z.real() + z.imag()*j. 
# I make my complaints here, this is an annoyance to everyone who isn't an electrical 
# engineer.
# But alas, this is what we have to deal with. 


get_ipython().run_line_magic('pylab', 'inline')
import numpy as np
import cmath
from pprint import pprint



class Clifford:
    def __init__(self,p,q):
        self.p = p
        self.q = q
        self.n = p+q
        self.k = self.n/2 if (self.n%2==0) else (self.n-1)/2
        self.matdim = int(pow(2,self.k))
        self.s = (q-p)%8
        self.ssp1 = self.s*(self.s+1)/2
        self.setup()

    """ get_chirality():
    
    Once we have constructed all of the clifford generators, they are stored in gens and
    are ordered so the hermitian generators first and the anti-hermitian generators are second
    
    We can then calculate the Chirality operator using: chirality = i^{s(s+1)/2} gamma^1 gamma^2 ... gamma^n
    """
    def get_chirality(self,p,q,gens):
        s = (q-p)%8
        chirality = complex(0,1)**(s*(s+1)/2)
        for gam in gens:
            chirality = np.dot(chirality,gam)
        return chirality
    
    
    
    """ introduce():
    
    This function prints out the type, the expected matrix dimension, the K0 dimension, the generators and the chirality operator. 
    This is useful to check that everything is working as planned. 
    TODO: Check that the generators satisfy the right anti-commutator relationships.
    
    """
    def introduce(self):
        print("My type is: ({},{})\n".format(self.p,self.q))
        print("My matrices will be {}x{}, and I have K0 dimension s = {}".format(self.matdim,self.matdim,self.s))
        
        """
        The type (1,3) Clifford module is used extensively in theoretical physics. It
        has a slightly different notation to the rest. It starts with gamma^0 for th hermitian operators
        and the three anti hermitian matrices being gamma^1, gamma^2 and gamma^3.
        
        All the other cases are start their labelling from 1 and go upwards.
        """
        if self.p==1 and self.q==3:
             # Print out the generators
            print("The generators are:")
            for i,gen in enumerate(self.generators):
                print("Gamma_{} = ".format(i)) # Starts the gamma label from zero
                print(gen)
            print("\n")
        else:
            # Print out the generators
            print("The generators are:")
            for i,gen in enumerate(self.generators):
                print("Gamma_{} = ".format(i+1)) # Starts the gamma labels from one. 
                pprint(gen)
            print("\n")

        print("The chirality operator is:")
        pprint(self.chirality)
    
    
    
    """ setup():
    
    This is the main function of the class which setups up the generators. 
    The small dimensional cases of (1,0), (0,1), (1,1), (2,0) and (0,2) are coded in by hand. 
    The higher dimensional cases are then constructed using the product mechanism that can be found in 
    Lawson and Michelsohn for instance, or there is a more to the point demonstration in my PhD thesis. 
    
    Note that for type (1,3) this does not produce the gamma matrices in the Dirac basis or the Chiral/Weyl basis or the 
    Majorana basis. 
    
    TODO: There is a function to apply a function that converts it to the Dirac/Chiral/Majorana basis
    """
    def setup(self):
        # Type (0,0) setup
        if self.p==0 and self.q==0:
            self.generators = []
            self.chirality = self.get_chirality(self.p,self.q,self.generators)
        
        # Type (0,1) setup
        elif self.p==0 and self.q==1:
            self.gamma1 = complex(0,1)
            self.generators = [self.gamma1]
            self.chirality = self.get_chirality(self.p,self.q,self.generators)
        
        # Type (1,0) setup
        elif self.p==1 and self.q==0:
            self.gamma1 = complex(1,0)
            self.generators = [self.gamma1]
            self.chirality = self.get_chirality(self.p,self.q,self.generators)
        
        # Type (0,2) setup
        elif self.p==0 and self.q==2:
            self.gamma1 = np.matrix([[complex(0,1),0],[0,complex(0,-1)]])
            self.gamma2 = np.matrix([[0,1],[-1,0]],dtype=np.complex)
            self.generators = [self.gamma1,self.gamma2]
            self.chirality = self.get_chirality(self.p,self.q,self.generators)

                
        # Type (1,1) setup
        elif self.p==1 and self.q==1:
            self.gamma1 = np.matrix([[1,0],[0,-1]])
            self.gamma2 = np.matrix([[0,1],[-1,0]])
            self.generators = [self.gamma1,self.gamma2]
            self.chirality = self.get_chirality(self.p,self.q,self.generators)

        # Type (2,0) setup
        elif self.p==2 and self.q==0:
            self.gamma1 = np.matrix([[1,0],[0,-1]])
            self.gamma2 = np.matrix([[0,1],[1,0]])
            self.generators = [self.gamma1,self.gamma2]
            self.chirality = self.get_chirality(self.p,self.q,self.generators)
        
#         # Type (1,3) setup - produces the Chiral representation.
#         # TODO: figure out how to get chiral and Dirac from the contruction below. 
#         elif self.p==1 and self.q==3:
#                 self.gamma0 = np.matrix([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]]) 
#                 self.gamma1 = np.matrix([[0,0,0,1],[0,0,1,0],[0,-1,0,0],[-1,0,0,0]])
#                 self.gamma2 = np.matrix([[0,0,0,complex(0,-1)],[0,0,complex(0,1),0],[0,complex(0,1),0,0],[complex(0,-1),0,0,0]])
#                 self.gamma3 = np.matrix([[0,0, 1,0],[0,0,0,-1],[ -1,0,0,0],[0, 1,0,0]])
#                 self.generators = [self.gamma0,self.gamma1,self.gamma2,self.gamma3]
#                 self.chirality = self.get_chirality(self.p,self.q,self.generators)
        
        # TODO: Procedure for calculating the other types from these two. See my thesis or textbook.            
        if self.n>2:
#         if self.n>2 and not (self.p==1 and self.q==3): # This line is a bit of an experiment. Type(1,3) has many forms that people use. 
            d_p = self.p//2 # Number of times to product with (2,0)
            d_q = self.q//2 # Number of times to product with (0,2) 
            r_p = self.p%2  # Number of times to product with (1,0) i.e. either once, or not at all
            r_q = self.q%2  # Number of times to product with (0,1) i.e. either once, or not at all
            
            cliff02 = Clifford(0,2)
            cliff20 = Clifford(2,0)
            cliff11 = Clifford(1,1)
            
            """
            There are four options:
                -  p = even and q = even
                -  p = even and q = odd
                -  p = odd and q = even
                -  p = odd and q = odd
            
            If p = even and q = even, Then start with (0,2) or (2,0) and we can product (2,0) and (0,2) together the correct number of times. 
            
            If p = even and q = odd, start with (0,1) and product with (2,0) or (0,2) the correct number of times
            
            If p = odd and q = even, start with (1,0) and product with (2,0) or (0,2) the correct number of times
            
            If p = odd and q = odd, start with (1,1) and product with (2,0) or (0,2)
            
            """
            
            """
            If p = even and q = even
            """
            if r_p==0 and r_q==0:
                if d_p!=0 and d_q==0:
                    input_gens = cliff20.generators
                    d_p = d_p-1 # Adjust for the fact we start from (2,0)
                    module_dim = 2
                if d_p==0 and d_q!=0:
                    input_gens = cliff02.generators
                    d_q = d_q-1 # Adjust for the fact we start from (0,2)
                    module_dim = 2
                if d_p!=0 and d_q!=0: # This branch needs thought
                    input_gens = cliff20.generators
                    d_p = d_p-1 # Adjust for the fact we start from (2,0)
                    module_dim = 2
                    
            # If p = even and q = odd
            elif r_p==0 and r_q==1: 
                input_gens = Clifford(0,1).generators
                module_dim = 1
                
            # If p = odd and q = even    
            elif r_p==1 and r_q==0: 
                input_gens = Clifford(1,0).generators
                module_dim = 1
                
            # If p = odd and q = odd
            elif r_p==1 and r_q==1: 
                input_gens = Clifford(1,1).generators
                module_dim=2


            """
            Products with (2,0) and (0,2)
            
            This procedure requires the first Clifford module to have even s. As both
            Cliff(2,0) and Cliff(0,2) have even s. This works.
            
            We will do the following
            let M = M(0,0)
                do M = M(2,0) x M  d_p times
                do M = M(0,2) x M  d_q times
            so that M = M(2*d_p, 2*d_q)
            """
            
            for i in range(d_p):
                holder_gens = []
                # This gives me the matrix dimension of the second module we are producting.
                for gen in cliff20.generators:
                    holder_gens.append(np.kron(gen,np.identity(module_dim)))
                for gen in input_gens:
                    holder_gens.append(np.kron(cliff20.chirality,gen))

                # After producting with a (2,0) the module_dim = matrix_dimension is increased by a factor of 2
                module_dim *= 2
                # Set the input_gens to be the just constructed generators so we can then repeat the process if necessary.
                input_gens = holder_gens

            for i in range(d_q):
                holder_gens = []
                # This gives me the matrix dimension of the second module we are producting.
                for gen in cliff02.generators:
                    holder_gens.append(np.kron(gen,np.identity(module_dim)))
                for gen in input_gens:
                    holder_gens.append(np.kron(cliff02.chirality,gen))

                # After producting with a (2,0) the module_dim = matrix_dimension is increased by a factor of 2
                module_dim *= 2
                # Set the input_gens to be the just constructed generators so we can then repeat the process if necessary.
                input_gens = holder_gens
            
            
            # Order the generators so the Hermitian elements are first, then the anti-Heritian
            # WARNING: The operator .H only works for np.matrix types, not np.array. 
            # The comparision, gen.H == gen, results in an array of True/False values. Then all(gen.H==gen) returns True if they are
            # all true and returns false if ANY of them are false. 
            herm = []
            anti_herm=[]
            for gen in input_gens:
                if all(gen.H==gen) ==False:
                    anti_herm.append(gen)
                elif all(gen.H==gen) == True:
                    herm.append(gen)
            
            # Set the generators to what has been calculated.     
            # The + operation here is action of Python lists, so it concatenates them. It doesn't
            # add the operators together. 
            self.generators = herm+anti_herm
            
            # Get the new chirality operator
            self.chirality = self.get_chirality(self.p,self.q,self.generators)
    """  ===== End of Setup ====  """





