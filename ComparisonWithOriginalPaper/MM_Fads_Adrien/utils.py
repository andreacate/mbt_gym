import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib
from tqdm import tqdm
from scipy.sparse import diags
from scipy.sparse.linalg import expm
from scipy.sparse.linalg import expm_multiply
from scipy import stats
from scipy.linalg import solve_banded
from scipy.integrate import quad





class mm_with_fads:
    """
    Model parameters for the environment.
    """

    def __init__(self,
                 k=2, phi=.00001, alpha=0.0001, 
                 q_bar=30, T=1, N_t=10000, N_u=1000, U_max=10,
                 sigma=1, eta=.1, qconst=0.6, mu=0,
                 Q_0 = 0, X_0 = 0, S_0=100, U_0 = 0, varphi=15, total_arrivals = 30, 
                 gamma = 1, matrixCJ = None):
        #psi = 14.87

        self.k = k
        self.phi = phi 
        self.alpha = alpha
        self.T = T
        self.sigma = sigma
        self.eta = eta
        self.qconst = qconst
        self.pconst = np.sqrt(1 - qconst ** 2)
        self.q_bar = q_bar
        self.mu = mu
        self.N_t = N_t
        self.delta_t = self.T / self.N_t
        self.t_vector = np.linspace(0, T, N_t)
        self.N_u = N_u
        self.U_max = U_max
        self.delta_u = 2*self.U_max / self.N_u

        self.q_vector = np.linspace(-q_bar  , q_bar, 2 * q_bar + 1) 
        self.u_vector = np.linspace(-U_max, U_max, N_u)
        self.Q_0 = Q_0
        self.U_0 = U_0
        self.S_0 = S_0
        self.X_0 = X_0
        self.gamma = gamma * self.qconst*self.sigma #it was just for me to avoid changing all the gamma in gamma *qconst * sigma 
        self.varphi = varphi 
        self.total_arrivals = total_arrivals
        self.aux_integral = self.compute_integral_related_to_psi() 
        self.psi = (self.total_arrivals * self.T - self.varphi*self.T) / self.aux_integral
        print("psi is ", self.psi)
        #self.psi = psi
        self.ode_result = self.ODE_solver()
        
        if matrixCJ is None:
            matrixCJ = self.get_delta_CJ_matrix()
            #print("matrixCJ computed", len(matrixCJ), matrixCJ[0].shape)
        self.matrixCJ = np.array(matrixCJ)

    def matrix_CJ(self): #use of sparse matrix because qconst may be large (since we assume 30 market arrivals on average)
        n = 2 * self.q_bar + 1

        main_diag = np.zeros(n)
        upper_diag = np.zeros(n - 1)
        lower_diag = np.zeros(n - 1)
        val = (self.varphi + self.psi * self.aux_integral/self.T) * np.exp(-1)
        for i in range(n):
            i_val = i - self.q_bar
            main_diag[i] = -self.k * self.phi * i_val ** 2
            if i < n - 1:
                upper_diag[i] = val
                lower_diag[i] = val

        M_sparse = diags(
            [lower_diag, main_diag, upper_diag],
            offsets=[-1, 0, 1],
            format="csc"
        )

        # for i in range(n):
        #     print(M_sparse[:,i].shape)
        #     qq=np.linspace(-self.q_bar,self.q_bar,n)
        #     print(qq.shape)
        #     print(np.array(M_sparse[:,i]))
        #     print((qq.T).shape)
        #     plt.plot(qq.T, np.array(M_sparse[:,i]))
        # plt.show
        return M_sparse   # matrix A 101*101

    def get_delta_CJ_matrix(self): 
        S = []
        #print("q_bar:", self.q_bar)
        n = 2 * self.q_bar + 1
        v = np.zeros(n)

        for i in range(n):
            i_val = i - self.q_bar
            v[i] = np.exp(self.k * (-self.alpha * i_val ** 2))


        M_sparse = self.matrix_CJ() 
        #print("M_sparse.shape:", M_sparse.shape)
        for t in self.t_vector:
            result = expm_multiply(M_sparse * (self.T - t), v) # result is omega pag 266 101*1
            #print("result.shape:", result.shape)
            S.append((1 / self.k) * np.log(result)) # S is h(t,q) pag 266
            #print("S[-1].shape:", S[-1].shape)
            #print("len S:", len(S))

        return S    # list of arrays, each array is h(t,q) 1000*101



    def get_delta_CJ_2(self, t, Q): #optimal displacements for CJP-strategy
        #print(self.matrixCJ.shape)
        matrixCJ = self.get_delta_CJ_matrix()
        self.matrixCJ = np.array(matrixCJ)
        D = 1 / self.k - self.matrixCJ[t][(self.q_bar + Q - 1).astype(int)] + self.matrixCJ[t][
            (self.q_bar + Q).astype(int)]
        G = 1 / self.k - self.matrixCJ[t][(self.q_bar + Q + 1).astype(int)] + self.matrixCJ[t][
            (self.q_bar + Q).astype(int)]
        return D,G # ask and bid

    def riccati_eq(self, t, P):
        term1 = -2 * (self.eta * P * (1 + self.qconst * (P * self.eta * self.qconst - self.qconst)))
        term2 = (1 + (P * self.eta * self.qconst - self.qconst) * self.qconst) ** 2
        term3 = (P * self.eta * self.qconst - self.qconst) ** 2 * self.pconst ** 2
        return term1 + term2 + term3
        

    def rk4_step(self, f, t, P, dt):
        k1 = f(t, P)
        k2 = f(t + 0.5 * dt, P + 0.5 * dt * k1)
        k3 = f(t + 0.5 * dt, P + 0.5 * dt * k2)
        k4 = f(t + dt, P + dt * k3)
        return P + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
       

    def riccati_solver(self): #computation of the Riccato equation
        P0 = 0.0
        t0 = 0.0
        dt = self.delta_t
        P_values = np.zeros(self.N_t)
        for i in range(self.N_t-1):
            P_values[i + 1] = self.rk4_step(self.riccati_eq, self.t_vector[i], P_values[i], dt)
        return P_values

    def ODE_solver(self,value_at_fundamental = False): #computation of the ODEs (with .__ for PI)
        def x_1(u):  
            return self.mu - self.sigma * self.qconst * self.eta * u

        x__2 = self.sigma  # functions x_2 and x_3
        x__3 = (-1) * self.riccati_solver() * self.qconst * self.eta + self.qconst * np.array(
            [1 for k in range(len(list(self.riccati_solver())))])
        A=np.zeros(self.N_t)
        A__=np.zeros(self.N_t)
        A__[-1] = - self.alpha
        b_0 = np.zeros(self.N_t)
        b_1 = np.zeros(self.N_t)
        A[-1] = - self.alpha
        c_0 = np.zeros(self.N_t)
        c_1 = np.zeros(self.N_t)
        c_2 = np.zeros(self.N_t)


        b_0__ = np.zeros(self.N_t)
        b_1__ = np.zeros(self.N_t)
        if value_at_fundamental == True:
            b_1[-1] = - self.qconst *self.sigma
            b_1__[-1] =- self.qconst *self.sigma
        else:
            b_1[-1] = b_1__[-1] = 0
        c_0__ = np.zeros(self.N_t)
        c_1__ = np.zeros(self.N_t)
        c_2__ = np.zeros(self.N_t)
        print("Adrien phi, varphi, psi, k, alpha", self.phi, self.varphi, self.psi, self.k, self.alpha)
        #print("Adrien self.delta_t:", self.delta_t)
        #print("Adrien self.N_t:", self.N_t)

        for t in range(self.N_t - 2, -1, -1):
            A[t] = A[t+1] + self.delta_t * (- self.phi + 4 * (self.psi+ self.varphi) * np.exp(-1) * self.k * A[t+1]**2)
            b_0[t] = b_0[t+1] + self.delta_t * (self.mu + 4 *self.k *(self.psi + self.varphi) * np.exp(-1) * A[t+1]*b_0[t+1])
            b_1[t] = b_1[t+1] + self.delta_t * (- self.eta * self.sigma * self.qconst - self.eta * b_1[t+1] + 4*(self.psi + self.varphi) * np.exp(-1)
                                                *A[t+1] *self.k*b_1[t+1] +4*np.exp(-1)*self.psi * self.gamma * A[t+1]
                                                +4*np.exp(-1)*self.k*self.gamma*self.psi*A[t+1]**2)
            c_2[t]=c_2[t+1] + self.delta_t * (- 2*self.eta*c_2[t+1] + (np.exp(-1)/self.k) * (self.k**2*(self.psi + self.varphi
                                                )*b_1[t+1]**2 + 2 * self.k**2 * self.psi *self.gamma * A[t+1]
                                                        *b_1[t+1] + 2*self.psi * self.k*self.gamma * b_1[t+1]))
            c_1[t] = c_1[t+1] + self.delta_t * (- self.eta * c_1[t+1] + np.exp(-1)/self.k * (2 * self.psi *self.k* self.gamma
                                                *b_0[t+1] + self.k**2*(self.varphi + self.psi) *2*b_0[t+1]*b_1[t+1]
                                                    +2*self.k**2 * self.psi*self.gamma*b_0[t+1]*A[t+1]))
            c_0[t] = c_0[t+1] + self.delta_t *(c_2[t+1] + (np.exp(-1)/self.k) *( 2*(self.psi + self.varphi) + 2*self.k*A[t+1]*(self.varphi+self.psi)
                                             + self.k**2 *(self.varphi+self.psi)*(A[t+1]**2 + b_0[t+1]**2)))




            A__[t] = A__[t + 1] + self.delta_t * (
                        - self.phi + 4 * (self.psi + self.varphi) * np.exp(-1) * self.k * A__[t + 1] ** 2)
            b_0__[t] = b_0__[t + 1] + self.delta_t * (
                        self.mu + 4 * self.k * (self.psi + self.varphi) * np.exp(-1) * A__[t + 1] * b_0__[t + 1])
            b_1__[t] = b_1__[t + 1] + self.delta_t * (- self.eta * self.sigma * self.qconst - self.eta * b_1__[t + 1] + 4 * (
                        self.psi + self.varphi) * np.exp(-1)
                      * A__[t + 1] * self.k * b_1__[t + 1] + 4 * np.exp(-1) * self.psi * self.gamma * A__[t + 1]
                        + 4 * np.exp(-1) * self.k * self.gamma * self.psi * A__[t + 1] ** 2)
            c_2__[t] = c_2__[t + 1] + self.delta_t * (
                        - 2 * self.eta * c_2__[t + 1] + (np.exp(-1) / self.k) * (self.k ** 2 * (self.psi + self.varphi
                            ) * b_1__[t + 1] ** 2 + 2 * self.k ** 2 * self.psi *
                                self.gamma *A__[t + 1] * b_1__[t + 1] + 2 * self.psi * self.k * self.gamma *b_1__[t + 1]))
            c_1__[t] = c_1__[t + 1] + self.delta_t * ( - self.eta * c_1__[t + 1] + np.exp(-1) / self.k *
                                                   (2 * self.psi * self.k * self.gamma* b_0__[t + 1] + self.k ** 2 *
                                                    (self.varphi + self.psi) * 2 * b_0__[ t + 1] * b_1__[t + 1] +
                                                    2 * self.k ** 2 * self.psi * self.gamma *b_0__[t + 1] * A__[t + 1]))
            c_0__[t] = c_0__[t + 1] + self.delta_t * (x__3[t+1]**2*c_2__[t + 1] + (np.exp(-1) / self.k) * (
                        2 * (self.psi + self.varphi) + 2 * self.k * A__[t + 1] * (self.varphi + self.psi)
                        + self.k ** 2 * (self.varphi + self.psi) * (A__[t + 1] ** 2 + b_0__[t + 1] ** 2)))

        return (A,A__,b_0,b_1,c_0,c_1,c_2,b_0__,b_1__,c_0__,c_1__,c_2__)
    

    def Value_fct_FI(self): #computation of the value function (useful to compute the following function)
        A,A__, b_0, b_1, c_0, c_1, c_2,b_0__,b_1__,c_0__,c_1__,c_2__ = self.ode_result
        H = np.zeros((self.N_t, len(self.u_vector),len(self.q_vector)))
        h =  np.zeros((self.N_t, len(self.u_vector),len(self.q_vector)))
        for u in range(len(self.u_vector)):
            for q in range(len(self.q_vector)):
                H[:,u,q] = self.q_vector[q]**2 * A +self.q_vector[q]*(b_0 + self.u_vector[u]*b_1) + (c_0 +  self.u_vector[u]*c_1 + self.u_vector[u]**2 * c_2)
                h[:,u,q] = self.q_vector[q]**2 * A__ +self.q_vector[q]*(b_0__ + self.u_vector[u]*b_1__) + (c_0__ +  self.u_vector[u]*c_1__ + self.u_vector[u]**2 * c_2__)
        return H,h


    def get_deltas_FI(self): #useful when you want to plot the optimal displacements (delta_a for the FI and delta_a__ for the PI), they look similar (because they are)
        H,h = self.Value_fct_FI()
        delta_a = np.zeros((self.N_t, len(self.u_vector),len(self.q_vector)))
        delta_b = np.zeros((self.N_t, len(self.u_vector),len(self.q_vector)))

        delta_a__ = np.zeros((self.N_t, len(self.u_vector),len(self.q_vector)))
        delta_b__ = np.zeros((self.N_t, len(self.u_vector),len(self.q_vector)))
        for u in range(len(self.u_vector)):
            for q in range(len(self.q_vector)-1):
                delta_b[:,u,q] = 1/self.k - H[:,u,q+1] + H[:,u,q]
                delta_b__[:,u,q] = 1/self.k - h[:,u,q+1] + h[:,u,q]
        for u in range(len(self.u_vector)):
            for q in range(1,len(self.q_vector)):
                delta_a[:, u, q] = 1 / self.k - H[:, u, q-1] + H[:, u, q]
                delta_a__[:, u, q] = 1 / self.k - h[:, u, q-1 ] + h[:, u, q]
        return delta_a,delta_b, delta_a__,delta_b__

    def get_delta_fast(self,t,u,q): #useful in the PnL function, we compute the value of the displacements for a precise value of (q,u,t)
        A, A__,b_0, b_1, c_0, c_1, c_2, b_0__, b_1__, c_0__, c_1__, c_2__ = self.ode_result
        #print("Adien A, b0, b1, c0, c1, c2:", A[t], b_0[t], b_1[t], c_0[t], c_1[t], c_2[t])
        
        h = q**2 * A[t] +q*(b_0[t] + u*b_1[t]) + (c_0[t] +  u*c_1[t] + u**2 * c_2[t])
        h_ = (q+1)**2 * A[t] +(q+1)*(b_0[t] + u*b_1[t]) + (c_0[t] +  u*c_1[t] + u**2 * c_2[t])
        h__ = (q-1)**2 * A[t] +(q-1)*(b_0[t] + u*b_1[t]) + (c_0[t] +  u*c_1[t] + u**2 * c_2[t])

        hat_h = q ** 2 * A__[t] +q * (b_0__[t] + u * b_1__[t]) + (c_0__[t] + u * c_1__[t] + u ** 2 * c_2__[t])
        hat_h_ = (q + 1) ** 2 * A__[t] + (q + 1) * (b_0__[t] + u * b_1__[t]) + (c_0__[t] + u * c_1__[t] + u ** 2 * c_2__[t])
        hat_h__ = (q - 1) ** 2 * A__[t] + (q - 1) * (b_0__[t] + u * b_1__[t]) + (c_0__[t] + u * c_1__[t] + u ** 2 * c_2__[t])

        #print("Adrien h, h_, h__:", h, h_, h__)

        delta_a =  1 / self.k - h__ + h
        delta_b =  1 / self.k - h_ + h

        delta_a_hat = 1 / self.k - hat_h__ + hat_h
        delta_b_hat = 1 / self.k - hat_h_ + hat_h
        return delta_a,delta_b, delta_a_hat,delta_b_hat

    def plotting(self): #useful if we want to plot U_t, U_t_hat, S_t (S_wt is the fundamental)
        S_t = np.zeros(len(self.t_vector))
        S_wt = np.zeros(len(self.t_vector))
        S_wt[0]= self.S_0
        S_t[0]= self.S_0
        U_t = np.zeros(len(self.t_vector))
        U_t_hat = np.zeros(len(self.t_vector))
        S_t_hat = np.zeros(len(self.t_vector))
        S_t_hat[0] = self.S_0
        P = self.riccati_solver()
        I= np.zeros(len(self.t_vector))
        for t in range(len(self.t_vector)-1):
            B = np.random.normal(0, 1, 1)
            Z = np.random.normal(0, 1, 1)
            S_t[t+1] = S_t[t] + (self.mu - self.eta * self.sigma * self.qconst * U_t[t]) * self.delta_t + self.sigma * (
                    self.qconst * np.sqrt(self.delta_t)*B + self.pconst * np.sqrt(self.delta_t)*Z)
            S_wt[t + 1] = S_wt[t] + (self.mu) * self.delta_t + self.pconst*self.sigma * np.sqrt(self.delta_t)*Z
            U_t[t+1] =U_t[t] - self.eta * U_t[t] * self.delta_t + np.sqrt(self.delta_t)*B
            dI_t = (S_t[t+1] - S_t[t] - (self.mu - self.eta * self.sigma * self.qconst * U_t_hat[t]) * self.delta_t) / self.sigma
            I[t+1] = I[t] + dI_t
            S_t_hat[t+1] =S_t_hat[t]+ (self.mu - self.eta * self.sigma*self.qconst*U_t_hat[t])*self.delta_t + dI_t*self.sigma
            U_t_hat[t+1] =U_t_hat[t] - self.eta * U_t_hat[t] * self.delta_t - (P[t] * self.qconst * self.eta - self.qconst) * dI_t

        return U_t,U_t_hat,S_t,S_t_hat,I,S_wt



    def PnL(self,sim, verbose = False, value_at_fundamental = False): #we compute the PnL following the 3 different strategies 
        PnL = np.zeros(sim)
        U_t = np.zeros(sim)
        Q_t = np.zeros(sim)
        S_t = np.zeros(sim)
        S_t= self.S_0*np.ones(sim)
        X_t = np.zeros(sim)

        PnL_hat = np.zeros(sim)
        U_t_hat = np.zeros(sim)
        Q_t_hat = np.zeros(sim)
        S_t_hat = np.zeros(sim)
        S_t_hat = self.S_0
        X_t_hat = np.zeros(sim)
        
        X_t_inf = np.zeros(sim)
        Q_t_inf = np.zeros(sim)
        X_t_uninf = np.zeros(sim)
        Q_t_uninf = np.zeros(sim)


        PnL_CJ = np.zeros(sim)
        X_t_CJ = np.zeros(sim)
        Q_t_CJ = np.zeros(sim)

        int_Q_t = np.zeros(sim)
        int_Q_t_inf = np.zeros(sim)
        int_Q_t_uninf = np.zeros(sim)
        int_Q_t_hat = np.zeros(sim)
        int_Q_t_CJ = np.zeros(sim)
        P = self.riccati_solver()
        
        PnL_inf = np.zeros(sim)
        PnL_uninf = np.zeros(sim)
        
        if verbose:
            Q = np.zeros((self.N_t, sim))
            U = np.zeros((self.N_t, sim))
        spread = []

        #tqdm

        for (it, t) in tqdm(enumerate(self.t_vector[:-1])):
            B = np.random.normal(0, 1, sim)
            Z = np.random.normal(0, 1, sim)
            ST = S_t.copy()
            S_t += (self.mu*np.ones(sim) - self.eta * self.sigma * self.qconst * U_t) * self.delta_t + self.sigma * (
                        self.qconst * np.sqrt(self.delta_t)*B + self.pconst * np.sqrt(self.delta_t)*Z)
            U_t += - self.eta * U_t * self.delta_t + np.sqrt(self.delta_t)*B
            dI_t = (S_t - ST - (self.mu*np.ones(sim) - self.eta*self.sigma*self.qconst*U_t_hat)*self.delta_t) / self.sigma
            U_t_hat+=- self.eta*U_t_hat*self.delta_t - (P[it] *self.qconst*self.eta - self.qconst)*dI_t



            delta_FI_a, delta_FI_b = self.get_delta_fast(it+1,U_t,Q_t)[:2] 
            delta_PI_a,delta_PI_b = self.get_delta_fast(it+1,U_t_hat,Q_t_hat)[2:]
            delta_a_CJ,delta_b_CJ = self.get_delta_CJ_2(it+1,Q_t_CJ)


            random_a = np.random.uniform(size = sim)
            random_b = np.random.uniform(size = sim)

            
            inf_a = self.psi * np.exp(-self.k * delta_FI_a - self.gamma * U_t) * self.delta_t
            uninf_a = self.varphi * np.exp(-self.k*delta_FI_a)* self.delta_t
            prop_inf_a = inf_a / (inf_a + uninf_a)
            inf_b = self.psi * np.exp(-self.k * delta_FI_b + self.gamma * U_t) * self.delta_t
            uninf_b = self.varphi * np.exp(-self.k*delta_FI_b)* self.delta_t
            prop_inf_b = inf_b / (inf_b + uninf_b)
            bern_a = np.random.binomial(1,prop_inf_a,size = sim)
            bern_b = np.random.binomial(1,prop_inf_b,size = sim)
            bern_a_un = 1-bern_a
            bern_b_un = 1-bern_b
            
            jump_a = random_a < (
                    self.varphi * np.exp(-self.k*delta_FI_a)* self.delta_t + self.psi * np.exp(-self.k * delta_FI_a - self.gamma * U_t) * self.delta_t)
            jump_b = random_b < (
                    self.varphi * np.exp(-self.k*delta_FI_b)* self.delta_t + self.psi * np.exp(-self.k * delta_FI_b + self.gamma * U_t) * self.delta_t)

            jump_a_hat = random_a < (
                    self.varphi * np.exp(-self.k*delta_PI_a)* self.delta_t + self.psi * np.exp(-self.k * delta_PI_a - self.gamma * U_t) * self.delta_t)
            jump_b_hat = random_b < (
                    self.varphi * np.exp(-self.k*delta_PI_b)* self.delta_t + self.psi * np.exp(-self.k * delta_PI_b + self.gamma * U_t) * self.delta_t)

            jump_a_CJ = random_a < (
                    self.varphi * np.exp(-self.k*delta_a_CJ)* self.delta_t + self.psi * np.exp(-self.k * delta_a_CJ - self.gamma * U_t) * self.delta_t)
            jump_b_CJ = random_b < (
                    self.varphi * np.exp(-self.k * delta_b_CJ)* self.delta_t + self.psi * np.exp(-self.k * delta_b_CJ + self.gamma * U_t) * self.delta_t)


            Q_t_inf += ((jump_a*bern_a).astype(int) - (jump_b*bern_b).astype(int))
            X_t_inf += -(S_t + delta_FI_a) * (jump_a*bern_a) + (S_t - delta_FI_b) * (jump_b*bern_b)
            
            Q_t_uninf += ((jump_a*bern_a_un).astype(int) - (jump_b*bern_b_un).astype(int))
            X_t_uninf += -(S_t + delta_FI_a) * (jump_a*bern_a_un) + (S_t - delta_FI_b) * (jump_b*bern_b_un)

            Q_t += (- jump_a.astype(int) + jump_b.astype(int))
            X_t += (S_t + delta_FI_a) * jump_a - (S_t - delta_FI_b) * jump_b

            Q_t_hat += (- jump_a_hat.astype(int) + jump_b_hat.astype(int))
            X_t_hat += (S_t + delta_PI_a) * jump_a_hat - (S_t - delta_PI_b) * jump_b_hat

            Q_t_CJ+= (- jump_a_CJ.astype(int) + jump_b_CJ.astype(int))
            X_t_CJ += (S_t + delta_a_CJ) * jump_a_CJ - (S_t - delta_b_CJ) * jump_b_CJ

            if verbose:
                Q[it +1, :] = Q_t
                U[it +1, :] = U_t

            int_Q_t += Q_t ** 2 * self.delta_t
            int_Q_t_inf += Q_t_inf ** 2 * self.delta_t
            int_Q_t_uninf += Q_t_uninf ** 2 * self.delta_t

            int_Q_t_hat += Q_t_hat ** 2 * self.delta_t
            int_Q_t_CJ +=Q_t_CJ ** 2 * self.delta_t
            spread.append(np.mean((delta_FI_b+delta_FI_a)/2))
        
        
        if value_at_fundamental:
            PnL = X_t + Q_t * (S_t - self.qconst * self.sigma * U_t)- self.alpha * Q_t ** 2 - self.phi * int_Q_t
            PnL_hat = X_t_hat + Q_t_hat *(S_t - self.qconst * self.sigma * U_t) - self.alpha * Q_t_hat ** 2 - self.phi * int_Q_t_hat
            PnL_CJ = X_t_CJ + Q_t_CJ * (S_t - self.qconst * self.sigma * U_t) - self.alpha * Q_t_CJ ** 2 - self.phi * int_Q_t_CJ
        else:
            PnL = X_t + Q_t * S_t- self.alpha * Q_t ** 2 - self.phi * int_Q_t
            PnL_hat = X_t_hat + Q_t_hat * S_t - self.alpha * Q_t_hat ** 2 - self.phi * int_Q_t_hat
            PnL_CJ = X_t_CJ + Q_t_CJ * S_t - self.alpha * Q_t_CJ ** 2 - self.phi * int_Q_t_CJ
            
            PnL_inf = X_t_inf + Q_t_inf*S_t - self.alpha * Q_t_inf ** 2 - self.phi * int_Q_t_inf
            PnL_uninf = X_t_uninf + Q_t_uninf*S_t - self.alpha * Q_t_uninf ** 2 - self.phi * int_Q_t_uninf
        
        if verbose:
            return PnL,PnL_CJ,PnL_hat,U,Q
        else:
            return PnL,PnL_CJ,PnL_hat, PnL_inf,PnL_uninf

    
    def PDE(self):
        V = np.zeros((self.N_t, len(self.q_vector), len(self.u_vector)))
        V[-1, :, :] = -self.alpha * self.q_vector[:, None] ** 2

        for t in range(self.N_t - 2, -1, -1):
            dV_du = np.zeros((len(self.q_vector), len(self.u_vector)))  # dV_du est 2D ici
            dV_duu = np.zeros((len(self.q_vector), len(self.u_vector)))  # dV_duu est aussi 2D

            # Différences finies centrées pour les dérivées en u (pour les indices internes)
            dV_du[1:-1, 1:-1] = (V[t+1, 1:-1, 2:] - V[t+1, 1:-1, :-2]) / (2 * self.delta_u)
            dV_duu[1:-1, 1:-1] = (V[t+1, 1:-1, 2:] + V[t+1, 1:-1, :-2] - 2 * V[t+1, 1:-1, 1:-1]) / (self.delta_u**2)

            # Extrapolation de Neumann : on impose que la dérivée est nulle aux bords
            dV_du[:, 0] = dV_du[:, 1]  # Extrapolation à gauche (bord inférieur)
            dV_du[:, -1] = dV_du[:, -2]  # Extrapolation à droite (bord supérieur)

            dV_duu[:, 0] = dV_duu[:, 1]  # Extrapolation à gauche pour la dérivée seconde
            dV_duu[:, -1] = dV_duu[:, -2]  # Extrapolation à droite pour la dérivée seconde

            # Optionnel : Lissage des dérivées pour éviter les pics
            dV_du[:, 1:-1] = (dV_du[:, 1:-1] + np.roll(dV_du[:, 1:-1], 1, axis=1) + np.roll(dV_du[:, 1:-1], -1, axis=1)) / 3
            dV_duu[:, 1:-1] = (dV_duu[:, 1:-1] + np.roll(dV_duu[:, 1:-1], 1, axis=1) + np.roll(dV_duu[:, 1:-1], -1, axis=1)) / 3

            # Mise à jour de V
            for q in range(1, len(self.q_vector) - 1):  # On évite les bords pour q
                q_val = self.q_vector[q]

                # Terme de drift
                term_1 = (self.mu - self.eta * self.sigma * self.qconst * self.u_vector) * q_val

                # **Limiter les termes exponentiels pour éviter l'explosion des valeurs**
                expo_term_1 = np.clip(self.k * (V[t+1, q-1, :] - V[t+1, q, :]), -50, 50)
                expo_term_2 = np.clip(self.k * (V[t+1, q+1, :] - V[t+1, q, :]), -50, 50)

                term_2 = (1 / self.k) * np.exp(np.clip(-1 + expo_term_1, -50, 50)) * (self.varphi + self.psi * np.exp(np.clip(-self.gamma * self.sigma * self.qconst * self.u_vector, -50, 50)))
                term_3 = (1 / self.k) * np.exp(np.clip(-1 + expo_term_2, -50, 50)) * (self.varphi + self.psi * np.exp(np.clip(self.gamma * self.sigma * self.qconst * self.u_vector, -50, 50)))

                # **Vérification de la stabilité des termes (contrôler les valeurs extrêmes)**
                # On peut aussi ajouter des limites sur term_2 et term_3 pour éviter une explosion
                term_2 = np.clip(term_2, -1e5, 1e5)
                term_3 = np.clip(term_3, -1e5, 1e5)

                # Schéma semi-implicite (résolution de système tridiagonal pour dV_duu)
                A_diag = (1 + self.delta_t / self.delta_u ** 2) * np.ones(len(self.u_vector))
                A_lower = -0.5 * self.delta_t / self.delta_u ** 2 * np.ones(len(self.u_vector) - 1)
                A_upper = -0.5 * self.delta_t / self.delta_u ** 2 * np.ones(len(self.u_vector) - 1)

                A = np.zeros((3, len(self.u_vector)))
                A[0, 1:] = A_upper  # Supérieur
                A[1, :] = A_diag  # Diagonal
                A[2, :-1] = A_lower  # Inférieur

                b = V[t+1, q, :] - self.delta_t * (self.eta * self.u_vector * dV_du[q, :] + self.phi * q_val ** 2 + term_1 + term_2 + term_3)

                # Résolution du système tridiagonal
                V[t, q, :] = solve_banded((1, 1), A, b)

        return V
    
    
    def compute_deltas(self):
        delta_a = np.zeros((self.N_t, len(self.q_vector),len(self.u_vector)))
        delta_b = np.zeros((self.N_t, len(self.q_vector),len(self.u_vector)))
        V = self.PDE()

        for u in range(len(self.u_vector)):
            for q in range(len(self.q_vector)-1):
                delta_b[:,q,u] = 1/self.k - V[:,q+1,u] + V[:,q,u]
        for u in range(len(self.u_vector)):
            for q in range(1,len(self.q_vector)):
                delta_a[:, q, u] = 1 / self.k - V[:, q-1, u] + V[:, q, u]
        return delta_a,delta_b


        

    def compute_integral_related_to_psi(self): #useful to rescale \psi, we do 150/compute_integral_related_to_psi() to get the value of \psi
        val = np.zeros(self.N_t)
        for t in range(len(val)):
            val[t] = np.exp(self.gamma**2 / 2 * (
                        (1 - np.exp(-2 * self.eta * self.t_vector[t])) / (2 * self.eta)))
        SUM = np.mean(val)
        return  self.T*SUM
    

    def plot_ODE_coefficients(self):
        """
        Plot the ODE coefficients returned by ODE_solver().
        Shows both the 'normal' and '__' (value at fundamental) solutions if present.
        """
        A,A__, b_0, b_1, c_0, c_1, c_2,b_0__,b_1__,c_0__,c_1__,c_2__ = self.ode_result
        time_to_maturity = self.time_grid if hasattr(self, "time_grid") else np.arange(len(A)) * self.delta_t
        #time_to_maturity = self.T - time_to_maturity  # plot as T - t

        fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
        fig.suptitle("ODE Coefficients vs Time-to-Maturity")

        # Row 1
        axs[0, 0].plot(time_to_maturity, A, label="A(t)")
        axs[0, 0].plot(time_to_maturity, A__, '--', label="A__(t)")
        axs[0, 0].set_title("A(t)")
        axs[0, 0].legend()

        axs[0, 1].plot(time_to_maturity, b_0, label="b0(t)")
        axs[0, 1].plot(time_to_maturity, b_0__, '--', label="b0__(t)")
        axs[0, 1].set_title("b0(t)")
        axs[0, 1].legend()

        # Row 2
        axs[1, 0].plot(time_to_maturity, b_1, label="b1(t)")
        axs[1, 0].plot(time_to_maturity, b_1__, '--', label="b1__(t)")
        axs[1, 0].set_title("b1(t)")
        axs[1, 0].legend()

        axs[1, 1].plot(time_to_maturity, c_0, label="c0(t)")
        axs[1, 1].plot(time_to_maturity, c_0__, '--', label="c0__(t)")
        axs[1, 1].set_title("c0(t)")
        axs[1, 1].legend()

        # Row 3
        axs[2, 0].plot(time_to_maturity, c_1, label="c1(t)")
        axs[2, 0].plot(time_to_maturity, c_1__, '--', label="c1__(t)")
        axs[2, 0].set_title("c1(t)")
        axs[2, 0].legend()

        axs[2, 1].plot(time_to_maturity, c_2, label="c2(t)")
        axs[2, 1].plot(time_to_maturity, c_2__, '--', label="c2__(t)")
        axs[2, 1].set_title("c2(t)")
        axs[2, 1].legend()

        for ax in axs.flat:
            ax.set_xlabel("Time t")
            ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_ODE_coefficient_differences(self):
        """
        Plot the differences between PI and FI ODE coefficients returned by ODE_solver().
        Shows the difference ('__' - 'normal') between value at fundamental solutions and regular solutions.
        """
        A, A__, b_0, b_1, c_0, c_1, c_2, b_0__, b_1__, c_0__, c_1__, c_2__ = self.ode_result
        time_to_maturity = self.time_grid if hasattr(self, "time_grid") else np.arange(len(A)) * self.delta_t
        time_to_maturity = self.T - time_to_maturity  # plot as T - t

        fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
        fig.suptitle("Difference between PI and FI ODE Coefficients vs Time-to-Maturity")

        # Calculate differences (PI - FI)
        diff_A = A__ - A
        diff_b0 = b_0__ - b_0
        diff_b1 = b_1__ - b_1
        diff_c0 = c_0__ - c_0
        diff_c1 = c_1__ - c_1
        diff_c2 = c_2__ - c_2

        # Row 1
        axs[0, 0].plot(time_to_maturity, diff_A, 'r-', label="A__(t) - A(t)")
        axs[0, 0].set_title("Difference in A(t)")
        axs[0, 0].legend()

        axs[0, 1].plot(time_to_maturity, diff_b0, 'g-', label="b0__(t) - b0(t)")
        axs[0, 1].set_title("Difference in b0(t)")
        axs[0, 1].legend()

        # Row 2
        axs[1, 0].plot(time_to_maturity, diff_b1, 'b-', label="b1__(t) - b1(t)")
        axs[1, 0].set_title("Difference in b1(t)")
        axs[1, 0].legend()

        axs[1, 1].plot(time_to_maturity, diff_c0, 'm-', label="c0__(t) - c0(t)")
        axs[1, 1].set_title("Difference in c0(t)")
        axs[1, 1].legend()

        # Row 3
        axs[2, 0].plot(time_to_maturity, diff_c1, 'c-', label="c1__(t) - c1(t)")
        axs[2, 0].set_title("Difference in c1(t)")
        axs[2, 0].legend()

        axs[2, 1].plot(time_to_maturity, diff_c2, 'y-', label="c2__(t) - c2(t)")
        axs[2, 1].set_title("Difference in c2(t)")
        axs[2, 1].legend()

        # Add a horizontal line at y=0 for reference
        for ax in axs.flat:
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.set_xlabel("Time to Maturity (T - t)")
            ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    # def plot_CJ_actions(self, sim=1000):

    #     Q_t_CJ = np.zeros(sim)
        
    #     for (it, t) in tqdm(enumerate(self.t_vector[:-1])):
    #         delta_a_CJ, delta_b_CJ = self.get_delta_CJ_2(it + 1, Q_t_CJ)

    #     Q_mid = len(self.q_vector) // 2
    #     u_mid = len(self.u_vector) // 2

    #     plt.figure(figsize=(12, 6))

    #     plt.subplot(1, 2, 1)
    #     plt.plot(time_to_maturity, delta_a_CJ[:, Q_mid, u_mid], label=f"delta_a (q={self.q_vector[Q_mid]}, u={self.u_vector[u_mid]})")
    #     plt.title("CJ Optimal Ask Quote Adjustment Over Time")
    #     plt.xlabel("Time to Maturity (T - t)")
    #     plt.ylabel("Delta Ask")
    #     plt.grid()
    #     plt.legend()

    #     plt.subplot(1, 2, 2)
    #     plt.plot(time_to_maturity, delta_b_CJ[:, Q_mid, u_mid], label=f"delta_b (q={self.q_vector[Q_mid]}, u={self.u_vector[u_mid]})", color='orange')
    #     plt.title("CJ Optimal Bid Quote Adjustment Over Time")
    #     plt.xlabel("Time to Maturity (T - t)")
    #     plt.ylabel("Delta Bid")
    #     plt.grid()
    #     plt.legend()

    #     plt.tight_layout()
    #     plt.show()
        
