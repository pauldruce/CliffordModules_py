#!/usr/bin/env python
# coding: utf-8

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

# In[352]:


get_ipython().system('jupyter nbconvert --to script CliffordAlgebras.ipynb')


# In[353]:


get_ipython().run_line_magic('pylab', 'inline')
import numpy as np
import cmath
from pprint import pprint
import sympy as sp


# In[431]:


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
        and the three anti hermitian matrices being gamma^1, gamma^2 and gamma^3..
        
        All the other cases are start their labelling from 1 and go upwards.
        """
        if self.p==1 and self.q==3:
             # Print out the generators
            print("The generators are:")
            for i,gen in enumerate(self.generators):
                print("Gamma_{} = ".format(i)) # Starts the gamma label from zero
                print(gen)
            print("\n")
        elif self.p==3 and self.q==0:
             # Print out the generators
            print("The generators are:")
            print("pauli.x = ") # Pauli Matrix = 1
            pprint(self.pauli_1)
            print("pauli.y = ") # Pauli Matrix = 2
            pprint(self.pauli_2)
            print("pauli.z = ") # Pauli Matrix = 3
            pprint(self.pauli_3)
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
    
        
        # TODO: Procedure for calculating the other types from these two. See my thesis or textbook.            
        if self.n>2:
#         if self.n>2 and not (self.p==1 and self.q==3):
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
            if self.p==3 and self.q==0:
                self.x = self.pauli_x = self.pauli_1 = pauli.generators[1]
                self.y = self.pauli_y = self.pauli_2 = -1*pauli.generators[2]
                self.z = self.pauli_z = self.pauli_3 = pauli.generators[0]
            # Type (1,3) setup - produces the Chiral representation.
            # TODO: figure out how to get chiral and Dirac from the contruction below. 
            elif self.p==1 and self.q==3:
                # Define the dirac basis for type (1,3)
                self.dirac0 = np.matrix([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]]) 
                self.dirac1 = np.matrix([[0,0,0,1],[0,0,1,0],[0,-1,0,0],[-1,0,0,0]])
                self.dirac2 = np.matrix([[0,0,0,complex(0,-1)],[0,0,complex(0,1),0],[0,complex(0,1),0,0],[complex(0,-1),0,0,0]])
                self.dirac3 = np.matrix([[0,0, 1,0],[0,0,0,-1],[ -1,0,0,0],[0, 1,0,0]])
                self.dirac_generators = [self.dirac0,self.dirac1,self.dirac2,self.dirac3]
                self.dirac_chirality = self.get_chirality(self.p,self.q,self.dirac_generators)
                
                # Define the majorana basis
                self.majorana0 = np.matrix([[0,0,0,complex(0,-1)],[0,0,complex(0,1),0],[0,complex(0,-1),0,0],[complex(0,1),0,0,0]])
                self.majorana1 = np.matrix([[complex(0,1),0,0,0],[0,complex(0,-1),0,0],[0,0,complex(0,1),0],[0,0,0,complex(0,-1)]])
                self.majorana2 = np.matrix([[0,0,0,complex(0,1)],[0,0,complex(0,-1),0],[0,complex(0,-1),0,0],[complex(0,1),0,0,0]])
                self.majorana3 = np.matrix([[0,complex(0,-1),0,0],[complex(0,-1),0,0,0],[0,0,0,complex(0,-1)],[0,0,complex(0,1),0]])
                self.majorana_generators = [self.majorana0,self.majorana1,self.majorana2,self.majorana3]
                self.majorana_chirality = -1*self.get_chirality(self.p,self.q,self.majorana_generators)
                
                # Define the Chiral basis
                self.chiral0 = np.bmat([[np.identity(2),np.zeros((2,2))],[np.zeros((2,2)),-1*np.identity(2)]])
                self.chiral1 = np.bmat([[np.zeros((2,2)), pauli.x],[-1*pauli.x,np.zeros((2,2))]])
                self.chiral2 = np.bmat([[np.zeros((2,2)), pauli.y],[-1*pauli.y,np.zeros((2,2))]])
                self.chiral3 = np.bmat([[np.zeros((2,2)), pauli.z],[-1*pauli.z,np.zeros((2,2))]])
                self.chiral_generators = [self.chiral0,self.chiral1,self.chiral2,self.chiral3]
                self.chiral_chirality= self.get_chirality(self.p,self.q,self.chiral_generators)
                
            # Set the generators to what has been calculated.     
            # The + operation here is action of Python lists, so it concatenates them. It doesn't
            # add the operators together. 
            self.generators = herm+anti_herm
            # Get the new chirality operator
            self.chirality = self.get_chirality(self.p,self.q,self.generators)
    """  ===== End of Setup ====  """


# In[432]:


cliff13 =Clifford(1,3)
cliff13.introduce()


# Check anti commutator relations for type (1,3)

# In[433]:


gamma0 = cliff13.generators[0]
gamma1 = cliff13.generators[1]
gamma2 = cliff13.generators[2]
gamma3 = cliff13.generators[3]
chirality13 = cliff13.chirality


# In[434]:


pprint(np.dot(gamma0,gamma0)+np.dot(gamma0,gamma0)==2*np.identity(4))
pprint(np.dot(gamma1,gamma1)+np.dot(gamma1,gamma1)==-2*np.identity(4))
pprint(np.dot(gamma2,gamma2)+np.dot(gamma2,gamma2)==-2*np.identity(4))
pprint(np.dot(gamma3,gamma3)+np.dot(gamma3,gamma3)==-2*np.identity(4))
pprint(chirality13*chirality13==np.identity(4))


# In[435]:


pauli = Clifford(3,0)


# In[436]:


pauli.introduce()


# In[438]:


U0 = np.bmat([[np.identity(2),np.identity(2)],[-1*np.identity(2),np.identity(2)]])
U1 = np.bmat([[np.identity(2),pauli.x],[-1*pauli.x,np.identity(2)]])
U11 = np.bmat([[np.identity(2),complex(0,1)*pauli.x],[-1*pauli.x,complex(0,-1)*np.identity(2)]])
U2 = np.bmat([[np.identity(2),pauli.y],[-1*pauli.y,np.identity(2)]])
U3 = np.bmat([[np.identity(2),pauli.z],[-1*pauli.z,np.identity(2)]])
U4 = np.bmat([[np.identity(2),pauli.chirality],[-1*pauli.chirality,np.identity(2)]])
# U = np.bmat([[a*pauli.x,complex(0,1)*b*pauli.z],[-1*complex(0,1)*b*pauli.z,a*pauli.x]])
# U = complex(0,1)*np.bmat([[np.identity(2),-1*pauli.x],[pauli.x,np.identity(2)]]).T
# U = np.bmat([[np.identity(2),np.identity(2)],[-1*np.identity(2),np.identity(2)]])
# U = np.bmat([[pauli.z,-1*pauli.y],[pauli.y,pauli.z]])

Us = [U0,U1,U2,U3,U4]


# In[442]:


for U in Us:
    new_gens = []
    for i,gen in enumerate(cliff13.generators):
        new_gens.append(U*gen*U.T/2)
        print("U * gamma{} * U.T = ".format(i))
        pprint(U*gen*U.T/2)
    print("U*chirality*U.T = ")
    pprint(U*cliff13.chirality*U.T/2)
    print("\n")


# In[443]:


V0 = np.bmat([[np.identity(2),np.identity(2)],[np.identity(2),-1*np.identity(2)]])
V1 = np.bmat([[pauli.x,np.identity(2),],[-1*np.identity(2), pauli.x]])
V2 = np.bmat([[pauli.y,np.identity(2)],[-1*np.identity(2),pauli.y]])
V3 = np.bmat([[pauli.z,np.identity(2)],[-1*np.identity(2),pauli.z]])
V4 = np.bmat([[pauli.chirality,np.identity(2),],[-1*np.identity(2),pauli.chirality]])


Vs = [V0,V1,V2,V3,V4]


# In[444]:


for U in Vs:
    new_gens = []
    for i,gen in enumerate(cliff13.generators):
        new_gens.append(U*gen*U.T/2)
        print("U * gamma{} * U.T = ".format(i))
        pprint(U*gen*U.T/2)
    print("U*chirality*U.T = ")
    pprint(U*cliff13.chirality*U.T/2)
    print("\n")

