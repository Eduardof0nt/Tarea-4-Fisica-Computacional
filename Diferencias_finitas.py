import numpy as np
import matplotlib.pyplot as plt

'''
Se define la función pxt(m,n,mat,params) donde:
    * m es la posición del arreglo que equivale al desplazamiento en 'x'
    * n es la posición del arreglo que equivale al desplazamiento en 't'
    * mat es la matriz de valores aproximados en la iteración actual
    * params es un dicionario con los parámetros que recibe la función (en este caso son D y delta)
'''
def pxt(m,n,mat,params):
    D = params['D']
    delta = params['delta']
    return (D*(mat[m+1,n]+mat[m-1,n])+delta*mat[m,n-1])/(2*D+delta)

'''
Se define la función GaussSeidel(mat, func, prec, maxIter, params) donde:
    * mat es la matriz de valores aproximados en la iteración actual
    * func es la función en la cual se calcula el resultado de cada punto de la malla
    * prec es la precisión que se desea alcanzar
    * maxIter es el número máximo de iteraciones antes de parar la ejecución
    * params es un dicionario con los parámetros que recibe la función (en este caso son D y delta)
'''
def GaussSeidel(mat, func, prec, maxIter, params):
    #Se extraen las dimensiones del arreglo
    s = mat.shape
    
    #Se inicia el contador de iteraciones
    iters = 0
    
    #Se inicia la variable que guarda si se llegó a la precisión máxima
    maxPrec = False
    
    #Guarda la precisión para retornarla al final de la ejecución
    difFinal = 0
    # Se inicia del ciclo de iteraciones para realizar las aproximaciones
    # El ciclo para si se llega a la precisión máxima o si se llega al número máximo de iteraciones
    while (iters < maxIter and not maxPrec):
        #Se agrega 1 al contador de iteraciones
        iters += 1
        
        for m in range(1,s[0]-1):
            for n in range(1,s[1]-1):
                #Valor anterior de la aproximación
                f0 = mat[m,n]
                
                #Se calcula el nuevo valor para el punto actual
                mat[m,n] = func(m,n,mat,params)
                
                #Se calcula la diferencia entre el valor nuevo y el anterior
                dif = np.abs((mat[m,n] - f0)/mat[m,n])
                if (dif < prec):
                    difFinal = dif
                    maxPrec = True

    #Se retorna como resultado la matriz con las aproximaciones, la precisión alcanzada y el número de iteraciones
    return (mat, difFinal, iters)

#Inicio del programa

#Se definen los parámetros para el cálculo que se desea realizar
params = {'D':0.5, 'puntosmalla': 100, 'Lx': 10, 'A': 2, 'x0': 5, 'l':1.5}

#Se define la malla de puntos y se calcula delta, que el la longitud dividida sobre la cantidad de puntos en la malla
puntosmalla = params['puntosmalla']
params['delta'] = params['Lx']/params['puntosmalla']
x = np.linspace(0, params['Lx'], puntosmalla)
t = np.linspace(0, params['Lx'], puntosmalla)
X, T = np.meshgrid(x, t)

#Se define la matriz inicial
pxti = np.zeros((puntosmalla, puntosmalla), float)

#Se introducen las condiciones iniciales en la matriz
for i in range(0, puntosmalla):
    pxti[0,i] = 0
    pxti[params['Lx'],i] = 0
    pxti[i,0] = params['A']*np.exp(-((params['delta']*i - params['x0'])**2)/params['l'])

#Se realiza el cálculo
(Z,prec,iters) = GaussSeidel(pxti, pxt, 0.000001, 1000, params)

#Se notifica al usuario sobre la precisión y el número de iteraciones
print('Precisión alcanzada:{0}\n Número de iteraciones:{1}'.format(prec,iters))
 
#Se grafican los resultados
plt.figure(figsize=(10,6))
ax = plt.axes(projection='3d')
ax.set_ylabel('x (m)')
ax.set_xlabel('t (s)')
ax.set_zlabel('p (u.c.)')
ax.plot_surface(X, T, Z, rstride=1, cstride=1,
            cmap='cividis', edgecolor='none')
ax.set_title('Aproximacion Difusión en una dimensión')
plt.show()
