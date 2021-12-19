# Title

## Zbiór danych treningowych
Zbiór danych treningowych składa się z 3402 zdjęć należących do 4 klas. W procesie uczenia dzielony jest z validation split 0.2  na podzbiory train (2722 zdjęć) i test (680 zdjęć).

1. Pine - 875 zdjęć (CL13\_Pinus\_muricata\_Bishop\_Pine\_OBJ)

![](./images/dataset/Pine.png)

2. Prunus - 850 zdjęć (FR16\_Prunus\_persica\_Peach\_Tree\_OBJ-png-white)

![](./images/dataset/Prunus.png)

3. Eucalyptus - 809 zdjęć (OC12\_Eucalyptus\_globulus\_Bluegum\_obj)

![](./images/dataset/Eucal.png)

4.Walnut  - 868 zdjęć (JA08\_Juglans\_ailantifolia\_Japanese\_Walnut\_OBJ)

![](./images/dataset/Walnut.png)

## Zbiór danych walidacyjnych 60-20-10-10
Zbiór danych walidacyjnych zawiera z 1075 zdjęć należących do 4 klas, każda z nich składa się z 3 podzbiorów.

Klasy:

1. Pine - 265 zdjęć (CL13\_Pinus\_muricata\_Bishop\_Pine\_OBJ)

![](./images/dataset_val/Pine60.gif)

2. Prunus - 269 zdjęć (FR16\_Prunus\_persica\_Peach\_Tree\_OBJ-png-white)

![](./images/dataset_val/Prunus60.gif)

3. Eucalyptus - 268 zdjęć (OC12\_Eucalyptus\_globulus\_Bluegum\_obj)

![](./images/dataset_val/Eucal60.gif)

4. Walnut  - 274 zdjęć (JA08\_Juglans\_ailantifolia\_Japanese\_Walnut\_OBJ)

![](./images/dataset_val/Walnut60.gif)

## Perceptron - porównanie Global<span style="color:blue">Average</span>Pooling i Global<span style="color:blue">Max</span>Pooling

### Multiperceptron - architektura

<div style="-webkit-column-count: 2; -moz-column-count: 2; column-count: 2; -webkit-column-rule: 1px dotted #e0e0e0; -moz-column-rule: 1px dotted #e0e0e0; column-rule: 1px dotted #e0e0e0;">
    <div style="display: inline-block;">
        <img src="images/perceptron/perceptron-40e.png" alt="">
    </div>
    <div style="display: inline-block;">
        <img src="images/perceptron-globalMAX/perceptron-40e-globalMAX.png" alt="">
    </div>
</div>

### Multiperceptron - kod

<div style="-webkit-column-count: 2; -moz-column-count: 2; column-count: 2; -webkit-column-rule: 1px dotted #e0e0e0; -moz-column-rule: 1px dotted #e0e0e0; column-rule: 1px dotted #e0e0e0;">
    <div style="display: inline-block;">
        <img src="images/perceptron/perceptron-code.PNG" alt="">
    </div>
    <div style="display: inline-block;">
        <img src="images/perceptron-globalMAX/perceptron-globalMAX-code.PNG" alt="">
    </div>
</div>

