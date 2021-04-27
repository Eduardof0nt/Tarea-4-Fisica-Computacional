# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:08:32 2021

@author: oscar
"""

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import scipy.integrate as integralscipy

#Se define la condición inicial para t=0 de la función de densidad, que solo dependería de x por lo que se crea una función
#Se opta por introducir de una vez las constantes proporcionadas por el ejercicio
def p0(x):
    '''
    Se define la condición de contorno para t=0

    Parámetros
    ----------
    x : Posición unidimensional tomada como el eje x.
    
    Salida de la función
    -------
    valorp0 : Retorna el valor de la densidad del material en una posición x

    '''
    valorp0 = 2.0*np.exp(-(x-5.0)**2/1.5)

    return valorp0

#Se crea la función que calcula cada coeficiente e_m
def E_m(longitud,m):
    '''
    Parámetros
    ----------
    longitud : representa la longitud máxima en x, pues es la frontera
    
    m : Representa el parámetro asociado al número de términos que tendrá la serie de Fourier, pues este
    coeficiente se debe calcular para cada iteración en particular. Cuando m es par, el coeficiente se anula,
    por lo que se opta por solo calcularlo para m impares

   Salida de la función
    -------
   Retorna el valor del coeficiente e_m

    '''
    def integrando(x):
        # Se crea una función para el integrando para que la función principal e_m reconozca la
        #función p0, pues esta condición es necesaria para calcular el parámetro e_m
        return p0(x)*np.sin(m*np.pi*x/longitud) 
    
    integral=integralscipy.quad(integrando,0,longitud) #Se utiliza la función quad de scipy para resolver la integral numéricamente
    
    valor_em=2.0*integral[0]/longitud #Se calcula el parámetro e_m con el resultado de la integral
                                      #La función quad de scipy retorna un arreglo con parámetros como el error, por lo que se coloca
                                      #el subíndice [0] para obtener solamente el resultado de la integral
    return valor_em

#Se crea la función que calcula la aproximación del valor de la densidad de la sustancia en cada punto x a través del tiempo
def Aprox_pXT(x, t, longitud, nt):
    '''
    Parámetros
    ----------
    x : posición en el eje x (o posiciones si se suministra un arreglo)
    
    t : momento en el tiempo (o momentos si se suministra un arreglo)
    
    longitud : lado del área donde se difumina la sustancia
    
    nt : número de términos que tendrá el cálculo de la densidad
    Salida de la función
    -------
    valorAprox_pXT : valor de la densidad de la sustancia en el punto (x, t)
    error: aproximación de la precisión mínima obtenida
    '''
    # Se inicializa el valor de la densidad a calcular
    valorAprox_pXT = 0

    # Se realiza la sumatoria que corresponde con el número de términos dado
    for n in range(0, nt):
        i=2*n+1 #Se calcula solo para los términos impares, pues para los pares se anula e_m
        
        #Se calcula la aproximación para cada iteración, añadiéndola en la variable establecida
        valorAprox_pXT += (E_m(longitud,i))*np.sin(i*np.pi*x/longitud)*np.exp(-(0.5*i**2*np.pi**2)/(longitud**2)*t)
        
        #Se crea una condición para calcular una aproximación de la precisión
        if i==2*(nt-1)+1:
            #Tomando el error como la diferencia entre la aproximación para n términos y n-1 términos, la diferencia corresponde
            #solo al último término de la aproximación para n términos
            dif=(E_m(longitud,i))*np.sin(i*np.pi*x/longitud)*np.exp(-(0.5*i**2*np.pi**2)/(longitud**2)*t)
            error=np.max(dif) #Se obtiene error máximo con la función de NumPy np.max del arreglo de diferencias
        else:
            pass

    return valorAprox_pXT , error  #Se retornan la aproximación y el error

#Se define la longitud de frontera establecida. No se establece límite para el tiempo, por lo que este queda
# a disposición del usuario para observar el comportamiento a través del tiempo

longitudespacial = 10
limittemporal = 20

# Se indica el número de términos para el cálculo de la aproximación del densidad de la sustancia
numeroterminos = 75

# Se define la malla de puntos para evaluar la densidad de la sustancia
puntosmalla = 30
x = np.linspace(0, longitudespacial, puntosmalla)
t = np.linspace(0, limittemporal , puntosmalla)
X, T = np.meshgrid(x, t)

# Se calcula el valor aproximado de la densidad de la sustancia en los puntos de  la malla y se asignan al eje Z
Z = Aprox_pXT(X, T, longitudespacial, numeroterminos)[0]

#Se le avisa al usuario la precisión mínima obtenida
precisión=Aprox_pXT(X, T, longitudespacial, numeroterminos)[1]
print("Se obtuvo un error de ", precisión)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('t (s)')
ax.set_ylabel('x (m)')
ax.set_zlabel('p (kg/m\xb3)')
ax.plot_surface(T, X, Z, rstride=1, cstride=1,
                cmap='cividis', edgecolor='none')
ax.set_title('Aproximación difusión en una dimensión');

#fig.savefig("Ecuación de difusión - Serie de Fourier.png")




