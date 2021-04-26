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
    coeficiente se debe calcular para cada iteración en particular

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
def Aprox_pXT(x, t, longitud, n):
    '''
    Parámetros
    ----------
    x : posición en el eje x (o posiciones si se suministra un arreglo)
    
    t : momento en el tiempo (o momentos si se suministra un arreglo)
    
    longitud : lado del área donde se difumina la sustancia
    
    n : número de términos que tendrá el cálculo del potencial (debe ser
           mayor o igual que 1)
    Salida de la función
    -------
    valorAprox_pXT : valor de la densidad de la sustancia en el punto (x, t)

    '''
    # Se inicializa el valor de la densidad a calcular
    valorAprox_pXT = 0

    # Se realiza la sumatoria que corresponde con el número de términos dado
    for i in range(1, n+1):
        
        #Se calcula la aproximación para cada iteración, añadiéndola en la variable establecida
        valorAprox_pXT += (E_m(longitud,i))*np.sin(i*np.pi*x/longitud)*np.exp(-(0.5*i**2*np.pi**2)/(longitud**2)*t)

    return valorAprox_pXT

#Se define la longitud de frontera establecida. No se establece límite para el tiempo, por lo que este queda
# a disposición del usuario para observar el comportamiento a través del tiempo

longitudespacial = 10
limittemporal = 20

# Se indica el número de términos para el cálculo de la aproximación del densidad de la sustancia
numeroterminos = 150

# Se define la malla de puntos para evaluar la densidad de la sustancia
puntosmalla = 30
x = np.linspace(0, longitudespacial, puntosmalla)
t = np.linspace(0, limittemporal , puntosmalla)
X, T = np.meshgrid(x, t)



# Se calcula el valor aproximado de la densidad de la sustancia en los puntos de  la malla y se asignan al eje Z
Z = Aprox_pXT(X, T, longitudespacial, numeroterminos)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('t (s)')
ax.set_ylabel('x (m)')
ax.set_zlabel('p (kg/m\xb3)')
ax.plot_surface(T, X, Z, rstride=1, cstride=1,
                cmap='cividis', edgecolor='none')
ax.set_title('Aproximación difusión en una dimensión');

#fig.savefig("Ecuación de difusión - Serie de Fourier.png")




