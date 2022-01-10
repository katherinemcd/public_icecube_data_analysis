{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import scipy \n",
    "import scipy.signal"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# we want to find B_i\n",
    "# we do this by scrambling data in a 6 degree elevation angle band in the sky\n",
    "# we can't use a histogram since there is spill over between bands (we want a sweep over the sky)\n",
    "\n",
    "# Load up the IceCube data\n",
    "icecube_data = np.load(\"./output_icecube_data.npz\", allow_pickle=True)\n",
    "data_sigmas = np.array(icecube_data[\"data_sigmas\"])\n",
    "data_ra = np.array(icecube_data[\"data_ra\"])\n",
    "data_dec = np.array(icecube_data[\"data_dec\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/software/python-anaconda-2020.02-el7-x86_64/lib/python3.7/site-packages/ipykernel_launcher.py:34: IntegrationWarning: The maximum number of subdivisions (50) has been achieved.\n",
      "  If increasing the limit yields no improvement it is advised to analyze \n",
      "  the integrand in order to determine the difficulties.  If the position of a \n",
      "  local difficulty can be determined (singularity, discontinuity) one will \n",
      "  probably gain from splitting up the interval and calling the integrator \n",
      "  on the subranges.  Perhaps a special-purpose integrator should be used.\n"
     ]
    }
   ],
   "source": [
    "# remove events with no reported errors\n",
    "data_ra = data_ra[data_sigmas != 0.0]\n",
    "data_dec = data_dec[data_sigmas != 0.0]\n",
    "data_sigmas = data_sigmas[data_sigmas != 0.0]\n",
    "\n",
    "# convert the dec to a sine of dec, turns out to be useful in that format.\n",
    "data_sin_dec = np.sin(data_dec)\n",
    "\n",
    "N = float(len(data_sigmas))\n",
    "\n",
    "# size of bins\n",
    "size_of_band = np.deg2rad(3.0)\n",
    "\n",
    "rad_ll = np.deg2rad(-87.0)\n",
    "rad_ul = np.deg2rad(87.0)\n",
    "\n",
    "# sweep over different sin decs to calculate the B_i at that point\n",
    "sweep_dec = np.linspace(rad_ll, rad_ul, 1000)\n",
    "\n",
    "sweep_counts = np.zeros(len(sweep_dec))\n",
    "for i in range(len(sweep_dec)):\n",
    "    solid_angle = 2.0 * np.pi * np.sin(size_of_band) * np.cos(sweep_dec[i])\n",
    "    entries_in_band = np.abs(data_dec - sweep_dec[i]) < size_of_band\n",
    "    total_in_band = np.sum(entries_in_band)\n",
    "    sweep_counts[i] += total_in_band / solid_angle\n",
    "    \n",
    "f_sweep = scipy.interpolate.interp1d(np.sin(sweep_dec),\n",
    "                                     sweep_counts,\n",
    "                                     kind = 'cubic',\n",
    "                                     bounds_error = False,\n",
    "                                     fill_value = 0.0)\n",
    "\n",
    "# to perform the average, integrate over result and divide it out\n",
    "sweep_counts_norm, err = scipy.integrate.quad(f_sweep, -1.0, 1.0) # to normalize\n",
    "\n",
    "# equation 2.2 in the paper\n",
    "P_B = sweep_counts / sweep_counts_norm\n",
    "B_i = P_B / (2.0 * np.pi)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "No handles with labels found to put in legend.\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYAAAAD8CAYAAAB+UHOxAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3dd3wU17XA8d9RR0ISTQghUUQvpotiTDFxA1xwN+41ghhiO4mdR+I4cZ5fnh3HJY2Yhx0cXHE32MZgTDAuFCPRZZroAiEJRBFN9bw/diCLENIu0mol7fl+PvvZnZl7Z86sVnt27ty5I6qKMcaYwBPk7wCMMcb4hyUAY4wJUJYAjDEmQFkCMMaYAGUJwBhjApQlAGOMCVAeJQARGS0im0QkU0SmVLD8dhFZ6zyWiEifquqKSDMRWSAiW5znpjWzS8YYYzxRZQIQkWBgKjAG6AHcKiI9yhXbDoxU1d7AU8B0D+pOARaqamdgoTNtjDGmlnhyBDAIyFTVbapaBMwCxrkXUNUlqnrQmVwGJHlQdxww03k9E7j2/HfDGGOMt0I8KJMI7HabzgIGV1L+fuBzD+rGq2o2gKpmi0jLilYmIqlAKkBUVNSAbt26eRCyMcaYU9LT0/eralz5+Z4kAKlgXoXjR4jIKFwJYJi3dc9FVafjNCmlpKRoWlqaN9WNMSbgicjOiuZ70gSUBbRxm04C9lawgd7AK8A4VT3gQd0cEUlw6iYAuR7EYowxpoZ4kgBWAJ1FJFlEwoDxwBz3AiLSFvgQuFNVN3tYdw5wt/P6bmD2+e+GMcYYb1XZBKSqJSIyGZgPBAMzVDVDRCY6y6cBvwWaA/8QEYASVU05V11n1c8A74rI/cAu4KYa3jdjjDGVkPo0HLSdAzDGBLri4mKysrI4efLkWcsiIiJISkoiNDT0jPkikq6qKeXLe3IS2BhjTB2RlZVFdHQ07du3x2lxAUBVOXDgAFlZWSQnJ3u0LhsKwhhj6pGTJ0/SvHnzM778AUSE5s2bV3hkcC6WAIwxpp4p/+Vf1fxzsQRgjDEByhKAMcYEKEsAxhhTz5yr96a3vTotARhjTD0SERHBgQMHzvqyP9ULKCIiwuN1WTdQY4ypR5KSksjKyiIvL++sZaeuA/CUJQBjjKlHQkNDPe7nXxVrAjLGmABlCcAYYwKUJQBjjAlQlgCMMSZAWQIwxpgAZQnAGGMClCUAY4wJUJYAjDEmQFkCMMaYAOVRAhCR0SKySUQyRWRKBcu7ichSESkUkUfd5ncVkdVujyMi8oiz7EkR2eO2bGzN7ZYxxpiqVDkUhIgEA1OBy4AsYIWIzFHVH9yK5QMPAde611XVTUBft/XsAT5yK/Kiqj5XrT0wxhhzXjw5AhgEZKrqNlUtAmYB49wLqGquqq4AiitZzyXAVlXded7RGmOMqTGeJIBEYLfbdJYzz1vjgbfLzZssImtFZIaIND2PdRpjjDlPniSAim4y6dVdB0QkDLgGeM9t9ktAR1xNRNnA8+eomyoiaSKSVtHwp8YYY86PJwkgC2jjNp0E7PVyO2OAlaqac2qGquaoaqmqlgEv42pqOouqTlfVFFVNiYuL83KzxhhjzsWTBLAC6Cwiyc4v+fHAHC+3cyvlmn9EJMFt8jpgvZfrNMYYUw1V9gJS1RIRmQzMB4KBGaqaISITneXTRKQVkAbEAGVOV88eqnpERCJx9SCaUG7Vz4pIX1zNSTsqWG6MMcaHxNubCPtTSkqKpqWl+TsMY4ypV0QkXVVTys+3K4GNMSZAWQIwxpgAZQnAGGMClCUAY4wJUJYAjDEmQFkCMMaYAGUJwBhjApQlAGOMCVCWAIwxJkBZAjDGmABlCcAYYwKUJQBjjAlQlgCMMSZAWQIwxpgAZQnAGGMClCUAY4wJUJYAjDEmQFkCMMaYAGUJwBhjApRHCUBERovIJhHJFJEpFSzvJiJLRaRQRB4tt2yHiKwTkdUikuY2v5mILBCRLc5z0+rvjjHGGE9VmQBEJBiYCowBegC3ikiPcsXygYeA586xmlGq2rfcTYmnAAtVtTOw0Jk2xhhTSzw5AhgEZKrqNlUtAmYB49wLqGquqq4Air3Y9jhgpvN6JnCtF3WNMcZUkycJIBHY7Tad5czzlAJfiEi6iKS6zY9X1WwA57llRZVFJFVE0kQkLS8vz4vNGmOMqYwnCUAqmKdebOMiVe2PqwlpkoiM8KIuqjpdVVNUNSUuLs6bqsYYYyrhSQLIAtq4TScBez3dgKrudZ5zgY9wNSkB5IhIAoDznOvpOo0xxlSfJwlgBdBZRJJFJAwYD8zxZOUiEiUi0adeA5cD653Fc4C7ndd3A7O9CdwYY0z1hFRVQFVLRGQyMB8IBmaoaoaITHSWTxORVkAaEAOUicgjuHoMtQA+EpFT23pLVec5q34GeFdE7gd2ATfV7K4ZY4ypjKh605zvXykpKZqWllZ1QWOMMaeJSHq5bviAXQlsjDEByxKAMcYEKEsAxhgToCwBGGNMgLIEYIwxAcoSgDHGBChLAMYYE6AsARhjTICyBGCMMQHKEoAxxgQoSwDGGBOgLAEYY0yAqnI0UGOMZzZkH+GRWasJCRb2HT5JfEwEg5KbcV2/RHq0jiE02H5vmbrFPpHG1ICyMuVn76wm6+BxIkKDGda5BU2jQpm1Yhfjpn7H7S8vp7i0zN9hGnMGOwIwphoOHitixnfb+SIjh005BTx/Ux9uGJB0xvIHXkvj+x35jHh2ER9Puoj4mAg/RmzMf9gRgDHnKa+gkEtfWMzf/p0JwLM39ub6/olnlGkaFcZr9w1iwsgO5BUUcs3fv+XIyWJ/hGvMWewIwBgvqSrPzNvIW8t2cbKklPcmXsjA9s3OWT4qPIRfjelOr8RYJr+1itEvfs1bPx5C+xZRtRi1MWezBGCMh7IOHmfSmyuJj4ngix9yuKhTcyaM6Fjpl7+7q3q3ZueB40z/ehvX/P1bZqVeSI/WMT6O2phz86gJSERGi8gmEckUkSkVLO8mIktFpFBEHnWb30ZEFonIBhHJEJGH3ZY9KSJ7RGS18xhbM7tkTM1RVZ6bv4lXvtnG/y3expqsw3zxQw7DO7fgtfsGM6JLnFfrmzSqE5/+dBghwUGM/es3pO3I91HkxlStyiMAEQkGpgKXAVnAChGZo6o/uBXLBx4Cri1XvQT4haquFJFoIF1EFrjVfVFVn6v2XhjjI5+szebvizJPT1/dpzUPX9KZ5BZRBAfJea2zTbNInr+pDxNeT+dXH65j/iMjCDrPdRlTHZ40AQ0CMlV1G4CIzALGAacTgKrmArkicqV7RVXNBrKd1wUisgFIdK9rTF2y68BxcgpO8s3mPF5btpNDx4vp0CKKiNBgBiU3Y8qYbkSEBld7O6O6teT5m/vw07dX8dLirdw9tD2Nw61F1tQuTz5xicBut+ksYLC3GxKR9kA/YLnb7MkicheQhutI4WAF9VKBVIC2bdt6u1ljqqSqPPreWlbtOsi2/cfOWNavbROm3taf1k0a1fh2x/ZKYOqiTP40f5PrcWNvbkppU+PbMeZcPEkAFR2bqjcbEZHGwAfAI6p6xJn9EvCUs66ngOeB+87akOp0YDpASkqKV9s1DcvmnAKiI0L4eNVe3ly+k7jocCaO7Eh8TAThIUF0T3CdUC0uLWPe+n10iY9mxY588goKubpPAh+u3EOX+Gjmrsvmqj6tWb/nMJ+s2UtMRCibcgoAuDkliYTYRjRvHMaVvRJoEhl23k09VQkOEmbeN4j/nbuB2av38sznGxnbK4EoOxIwtcSTT1oW4P6zJAnY6+kGRCQU15f/m6r64an5qprjVuZl4FNP12kCg6qyavchysqU15buZM6aMz92BSdLmPB6+hnzRvdsxdHCEr7N3H/G/L8s3HLG9Bc/uD5+PRJi+CH7CN0TYvjwJ0NpFFb95h1vxMdE8Jfx/bh7aHuu/8cS3ly+k9QRHWs1BhO4PEkAK4DOIpIM7AHGA7d5snIREeCfwAZVfaHcsgTnHAHAdcB6j6M2Dd76PYf585eb+XJD7hnzO8ZF8Zsre9C/bVMkCJ6dt5HIsBCmf70NgHkZ+wC4tHtLwkOCubBjc37zseuj9dYDg/l49R4u7NicfyzaylW9W/PQJZ3ILSikWVSYX8fq6d+2KYOSm/Ha0p3cP6yDz446jHEnqlW3qjhdNP8MBAMzVPUPIjIRQFWniUgrXO34MUAZcBToAfQGvgHWOfMBfq2qc0XkdaAvriagHcAEt4RQoZSUFE1LS/N6J0398co323jpq63kHy8iLDiIS7vHk3e0kMu6x/PjER3OWW9LTgELN+Zy2+C2aBnERoaeXjZnzV4y9hzmV2O718YunLe567J58M2V/PaqHtw3LNnf4ZgGRETSVTXlrPmeJIC6whJAw/btlv3c8c/ldE+I4ZJuLbn3ovY0bxzu77BqTUlpGRNeT2fhxlzeSR3C4A7N/R2SaSDOlQBsLCBTJ+zOP85P3nS15//5lr48ekXXgPryBwgJDmLq7f1pEhnKX/+9haISGz3U+JYlAON3xwpLuHHaEgpLyvj3L0bStVW0v0Pym4jQYB69vCvfZR7ghpeWcPi4DRxnfMcSgPGr4tIyLnl+MTlHCnnh5j50iGvs75D87o4h7Zh2xwA27jvCw++sorSs/jTTmvrFEoDxq3dW7GbfkZPcMaQtV/ZK8Hc4dcboC1rxu6t78tWmPB57bw2ZuUf9HZJpgCwBGL8pLi3jpa+2ktKuKU+NuwBXr2Fzyu2D2zJ5VCc+Wr2Hq/72DQs35FBg9xIwNcgSgPGb385ez55DJ3hgeAf78q+AiPDoFV358ucjaRUTwf0z0xj13GJyC076OzTTQFgCMH4xbfFW3v5+N/cMbc8VPeP9HU6d1jGuMa/dN5hhnVqw/2gh0xdv83dIpoGwBGBq3fo9h3lu/iZG92zF41d2t1//HmjbPJI3HhjM9f0SeXP5LusdZGqEJQBTqw4eK+Jn76ymeeMw/nhDb78Ov1Af3TcsmRPFpTwxe72dDzDVZv99ptYUl5Zx47QlbMk9yh9v6H3GcA3GMz1bx5A6ogOfrcvm6c83+jscU89ZAjC1oqxMeWBmGlvzjvHYFV25uGtLf4dUL4kIvx7bnVsHteG9tN1kHTzu75BMPWYJwNSKOWv2snhzHj8ensyESgZ1M56ZNKoTgjDV7XaVxnjLEoDxOVXl3bTdtGnWiF+P7U6ItftXW0JsI24b3Jb30rJYtu2Av8Mx9ZT9JxqfKiwp5bH317Jk6wFuH9zOevzUoEmjOpHYtBHjpy/jvz+x22wb71kCMD71yjfbeT89i1sHteXHw63ppybFRYfz0YMXMaxTC2Z8t50XF2ymzMYNMl6wBGB85vCJYv7y5Rau6BnP09f3srtc+UCzqDCm3zWAvm2a8JeFW5j01kpyC05Sn+7zYfzHEoDxiSMni/n9JxkUlZYxYaTd49aXIsNC+OjBoTx2RVc+X7+PQX9YyHX/WMLRwhJ/h2bqOE/uCWyMV0qdLp/fb8/n+v6J9E1q4u+QGjwRYdKoTqgqK3Yc5JstefxudgZ/urE3QXbkZc7BoyMAERktIptEJFNEplSwvJuILBWRQhF51JO6ItJMRBaIyBbnuWn1d8fUBdO/3sb32/P54w29eOHmvvYFVIsm/6gzM+8bxKRRnfhgZRb3/GsFJ4tL/R2WqaOqTAAiEgxMBcbgutH7rSLSo1yxfOAh4Dkv6k4BFqpqZ2ChM23quV+8u4Y/zttI1/horu+f5O9wAtbPL+vCU9dewNeb83js/bV2cthUyJMjgEFApqpuU9UiYBYwzr2Aquaq6gqg/OAkldUdB8x0Xs8Erj3PfTB1QFFJGU/OyeCDlVkMbN+UjyddZOP8+JGIcOeQdjx2RVc+WbOXxz9ex4kiOxIwZ/LkHEAisNttOgsY7OH6K6sbr6rZAKqaLSIVjg0gIqlAKkDbtm093KypTfuPFvLgGyv5fkc+XeIb8+ItfWkUFuzvsAzw4MUdOXyimOlfb2N3/gmu6BlPWEgQV/dpTWSYnQIMdJ58AipqwPX0eLI6dV2FVacD0wFSUlLsOLaOWbJ1P7e9vByA31zZnQesr3+dcmrsoJJSZcZ32/k2cz8Av52dQVhIECO6xPHCzX0ID7GEHYg8SQBZQBu36SRgr4frr6xujogkOL/+E4BcD9dp6ghV5U/zNwHwxFU9uHdoe/8GZM7pV2O70SI6jIHtm3GyuJQ3lu1EFT5bm02zyDCeuvYCf4do/MCTBLAC6CwiycAeYDxwm4frr6zuHOBu4BnnebYXcZs6YPHmPFbtOsQfrruA2we383c4phKhwUE8eHGn09PDO8cB8PtPMvjXkh3cNrgt3RNi/BWe8ZMqz9KpagkwGZgPbADeVdUMEZkoIhMBRKSViGQBPwd+IyJZIhJzrrrOqp8BLhORLcBlzrSpJ1buOsg9r64gsUkjbhrQpuoKpk565JIuRIeH8Md5dm+BQOTRWSBVnQvMLTdvmtvrfbiadzyq68w/AFziTbCmbjh4rIhH310DwDM39CIsxHr71FexkaFMGtWJpz/fyOfrshnTK8HfIZlaZP+5xivFpWXc8NISdh88zjupQ043JZj66+6h7ekQF8VP3lzJT99excFjRf4OydQSSwDGK19k5LBt/zH+cF0vBndo7u9wTA2ICA3m40kXMa5vaz5Zs5cJb6SzLe+ov8MytcASgPHYd5n7+dm7q0luEcX1/RL9HY6pQTERofxlfD+mjOnG99vzuepv35K+M9/fYRkfswRgPLLn0AnufXUFrWMjePWegXZXrwZq4siOvHbfIMpUeXjWap7+fANfZOzzd1jGR+y/2FTpeFEJv/pwHYryxgODad8iyt8hGR8a0SWONx8YQm5BIf+3eBupr6fz7405/g7L+IAlAFOpktIyJryezteb80gd0YGkppH+DsnUggHtmvLNL0fxqzHd6BLfmAmvp7PVzgs0OJYATKWmLd7KN1v28z/XXsBjV3TzdzimFsXHRDBhZEfefGAIESHB/OrDdRSVlPk7LFODLAGYcyouLeO1pTsZ2SWOO4bYlb6BKi46nN+P68n32/P5+burKbWhpRsMGw7QnNNLX20lt6CQZ26wL/9Ad33/JNdn4fON7D10gtfvH0xUuH191Hd2BGAqtO/wSV7+ZhuXdGvJqK4VjtRtAszEkR25qFNzVu46xC8/WGs3nm8ALIWbCv1uznqKS8t45NIuiNgtHY3Li7f05c5Xvueztdl8tjab3kmx7Mo/TnKLKN6wo4J6x44AzFnW7znM/IwcJo7sSK+kWH+HY+qQltERzHtkOKN7tqJjXBTHCks4dLyYVbsO8dO3V9n5gXrG0rU5ywsLNhPbKJT7hiX7OxRTB4kI0+4ccHpaVXnq0w3M+G47E15PY1ByMy5IjGVoxxZ+jNJ4whKAOcPu/OP8e2MuP7+sCzERof4Ox9QDIsITV3Vn474jfLkhly83uO7tNP3OAVzes5WfozOVsQRgznDqloFjbVhg4wURYcY9A9mSc5QN2Ud4YcFmfv3ROoZ1bmH3Hq7D7ByAOcOc1Xtp2yySjnE23IPxTkRoML2SYrl5YBum3t6P/UeLuPpv3/L60h3szj9uVxLXQZYAzGk7Dxxj6bYD3JySZD1/TLUMaNeMF27uQ2hwEE/MzmD4s4u45PnFfOccYZq6wRKAOe29tCxE4IYBFd7czRivXN8/iY8evIi/jO/L5FGu+xHf/spynvr0Bz9HZk7xKAGIyGgR2SQimSIypYLlIiJ/dZavFZH+zvyuIrLa7XFERB5xlj0pInvclo2t2V0z3igsKWXWit1c3CWOhNhG/g7HNBCNwoIZ1zeRR6/oyqc/HQbAP7/dTtoOu9dAXVBlAhCRYGAqMAboAdwqIj3KFRsDdHYeqcBLAKq6SVX7qmpfYABwHPjIrd6Lp5Y79w42fjJ3XTb7jxZyz0XW9dP4xgWJsax84jLiosO5/ZXlLN92wN8hBTxPjgAGAZmquk1Vi4BZwLhyZcYBr6nLMqCJiJTvRnIJsFVVd1Y7alOjSsuUl7/eTse4KEZ0tr7bxneaRYXx0YNDaR4VxqS3VnGiqNTfIQU0TxJAIrDbbTrLmedtmfHA2+XmTXaajGaISNOKNi4iqSKSJiJpeXl5HoRrvPXqd9v5IfsID13S2U7+Gp9LahrJMzf0Zv/RQoY/u4hJb63kiY9dQ4+Y2uVJAqjoG6H89d6VlhGRMOAa4D235S8BHYG+QDbwfEUbV9XpqpqiqilxcXEehGu8oarMXLqDCzs055o+rf0djgkQI7rE8eTVPQgSmL9+H68v28mLCzb7O6yA40kCyALauE0nAXu9LDMGWKmqp+8rp6o5qlqqqmXAy7iamkwtm716L7vzT3DzQOv6aWrXPRcl8/3jl5L5v2O5JaUN0xZvJTO3wN9hBRRPEsAKoLOIJDu/5McDc8qVmQPc5fQGGgIcVtVst+W3Uq75p9w5guuA9V5Hb6ol/1gRv/l4Pf3aNuHq3vbr3/jPL0d3JSQ4iBunLeXOfy7nL19uYdM+Swa+VmUCUNUSYDIwH9gAvKuqGSIyUUQmOsXmAtuATFy/5h88VV9EIoHLgA/LrfpZEVknImuBUcDPqrszxjtf/pDD0cISnhp3ASHBdkmI8Z/mjcP52639aNIolG+27OfFLzcz6a2VNrqoj3k0SIfTRXNuuXnT3F4rMOkcdY8DzSuYf6dXkZoa923mfuKiw+nZOsbfoRjDFT1bcXmPeErLlJlLd/LUpz/wj0WZpI7sQEhQEMFBQlFJGWEh9mOlptgoTQHq8IlivtqUy4+6tbS2f1NniAghwcI9Q9uzeHMezy/YzPPOyeG46HDyjxXxs0s7M/lHnf0cacNgqTRAvfrddo6cLOHmgW2qLmxMLQsOEp65vhdX92lNUlPXlentmkXSKzGW577YzJ/mb2TXgeN+jrL+syOAADVv/T4Gtm9qN+0wdVbrJo342639zphXXFrGT95IZ+qirUxdtBWAnq1j6NYqhst7xtM7KZZWMRFnHNWu2nWQXfnHWbghlzEXtGJQcjPAdVHaq9/toHN8Y4Z1ahGQR8KWAALQzgPH2LivgN9c2d3foRjjldDgIF65eyAZew/zi3fXsO/ISfKPFfHByiw+WJkFgAj0Sozl/mHJpO88yGtL/zP4wJw1/+mdHhYcRJHbxWcz7xvEyC6Bda2RJYAAND9jH+A66WZMfdSzdSzzHhlxenrXgeOs3XOI3fknmJexjzW7D/HwrNUADGjXlN9e1YMmkaF8ti6bXQeOk1dQSJPIMMJDg8g+dIJFm/K4718reHfChQxoV+GgBA2SuDrw1A8pKSmalpbm7zDqvRteWsKJolLmPjzc36EY4xPr9xzm6y15xEdHMK5v6yq7ORecLGbUc1/RPSGG1+8fXEtR1h4RSVfVlPLz7SRwgNl76ATpOw8y+gL79W8argsSY3nw4k7cMCDJo2tcoiNCSR3RgW+27OfOfy5n+/5jrN9zmGmLt1LSgMcosiagAPPpWlcbqI37Y8yZ7rqwPR+t2ss3W/Yz6rmvTs9/5vON3JLShqev70VQUMM6UWxHAAHm8/X76J0US/sWds9fY9xFhAbz+cPDeeuBwURHhNCtVTQ9EmIY0qEZ76TtZtD/LuSD9Cx/h1mj7BxAACk4WUzf/17AxJEdeOyKbv4Ox5h6QVW56m/fkrH3CADhIUH8eHgHfjy8A7GRoX6OzjN2DsDw6dpsSsuUH3WL93coxtQbIsK/7h1E6ogOxMeEE9solL8vyiTlDwvq/eildgQQIE4Wl3LJ84tpGhXKJ5OHBeRFL8bUBFXljeW7eOJj1wDG/3PtBdwxpJ2fo6qcHQEEuDeW7WTPoRNMGd3dvvyNqQYR4Y7BbfnzLX1JbhHFn7/czNHCkjPKnBrFNPfISe6a8T3L6uj9j60XUAA4UVTK1EWZDO/cgmF2z19jqk1EuLZfIolNG3HTtKVc8Lv5dIyLoqi0jN35JwDo1LIxAmzJPcr6PYeZlTqELvHR/g28HEsAAeCTtXs5eLyYSaM6+TsUYxqUge2b0allYzJzj7I179gZyzJzj55+nX+siMtf/Jou8Y3pnhDD0I7NuWVg29oO9yyWABo4VeW1pTvo1LIxg51BsIwxNWf+IyPYceAYsY1COVlcSuvYRpSqsjbrEPMzcrhjcDtG/GkRAJtzjrI55yizV++lS3w0/dr6d9gJOwfQwM1dt4/1e46QOryDtf0b4wPBQULHuMa0aBxOUtNIgoKE0OAgBrRrxq/Hdqdt80jmPjScF27uc8Y4Qy849znwJzsCaMCOnCxmyodr6RLfmOv6J/o7HGMCVo/WMfRoHcP1/ZMAeOWbbfzPZxtYkrmfXkmxREf453oCj44ARGS0iGwSkUwRmVLBchGRvzrL14pIf7dlO5x7/64WkTS3+c1EZIGIbHGeA2cIvlqgqjw9dyMFJ0t44ea+hNo9f42pM+4Y0o7YRqHc9spyej35BSeKSv0SR5XfCiISDEwFxgA9gFtFpEe5YmOAzs4jFXip3PJRqtq3XD/UKcBCVe0MLHSmTQ35dG02b3+/i5tTkrggMdbf4Rhj3ESEBvPTH/2nU8aSrfv9EocnPwsHAZmquk1Vi4BZwLhyZcYBr6nLMqCJiCRUsd5xwEzn9UzgWi/iNpVYv+cwj3+0jm6tonn6+t7+DscYU4EHhndg0aMXA3D/zLTT1w7UJk8SQCKw2206y5nnaRkFvhCRdBFJdSsTr6rZAM5zy4o2LiKpIpImIml5eXkehBvYFm/O47p/fMeRkyX815huBDew0QuNaUiSW0TRtlkkANMWb6317XuSACr6Bimfqiorc5Gq9sfVTDRJREZUUPacVHW6qqaoakpcXGDdrs1bZWXK03M30DwqnLceGMyorhXmVGNMHfLZQ8MAWJt1qNa37UkCyALauE0nAXs9LaOqp55zgY9wNSkB5JxqJnKec70N3pzp/fQsNu4r4L/GdGVoJ7vi15j6IDoilFtS2jA/I4f/en9trW7bkwSwAugsIskiEgaMB+aUKzMHuMvpDTQEOKyq2SISJSLRACISBVwOrHerc7fz+m5gdn2K7rEAAA5vSURBVDX3JeBs33+MD9KzeGfFLn769ip++cFaBrVvxrV9rcunMfXJXUNdg8m9k7abjfuO1Np2q7wOQFVLRGQyMB8IBmaoaoaITHSWTwPmAmOBTOA4cK9TPR74yLkAKQR4S1XnOcueAd4VkfuBXcBNNbZXDVRZmRIUJGzNO8qs73fxyrfbcR/MtWt8NE/f0Msu+DKmnunZOpa/3dqPn769im+37Kdbq5ha2a4NB10PLNm6nwmvp1NwsoQWjcPZf7SQIIGRXeJ4YHgHThaXMii5md8uJjHG1Iz2Uz4DYMczV9boes81HLRdCVyHLdm6n/9bvI0lW/dTXKrcNCCJ48WlhIcE8csrutEqNsLfIRpjfOD99CxuHJDk8+1YAqijMvYe5raXlwPQrVU0s1KH0CQyzM9RGWN86dV7B3Lvqyt49L01lgAC1dqsQ9z+ynJaNA7nr7f2ZUhyc4KsP78xDZ571+2ikjLCQnw7hIsNEFPHvL50B9f8/TvKypR/3TuQoR1b2Je/MQHouS82+XwblgDqkH9vzOGJ2RkkNW3E26lDbAwfYwLQFz9zXSs7/ettPt+WJYA64v30LO77VxoJsRF8+fOR9E5q4u+QjDF+4H7byHnr9+HLnpqWAOqA15ft5NcfrqNf2ya8fFcKEaHB/g7JGONHp8YHmvhGOu+s2F1F6fNnCcDPPlubzRMfr6dd80hm3D3Qmn2MMbydOuT066+3+G4QTEsAfrI17yh3z/ien7+7mj5tmjD34eE0jbJunsYYSGzSiN9d7brtyv6CIp9tx7qB+sH76Vk8+t4aADq0iOLvt/azO3YZY85w70XJbMk9yrz1+3y2DUsAPqaqvLFsJ+GhwRScLOHN5TvZlneMIIEpY7qROqKjv0M0xtRRW3OPkn+siK835zGiS80Ph28JwAfSd+bz4oIt7Dl0gu37j52xLDRYuL5/In+4theNwuxkrzHm3IpKywC4a8b3NT4+EFgCqFFlZcqn67J5/KN1FJaUUVTi+uNFhQUzvHMc9w1LJqVdU7uwyxjjkXcnXEjnxz/32fotAdSQzNwCfjs7gyVbD9A6NoJ5jwwl/2gRJWVl9G3TxIZoNsZ4LTQ4iE4tG9M00jcj/VoCqAHb9x/j2qlLKCot4/fX9OTGAUlEhYeQ2KSRv0MzxtRzCbERFJws8cm6LQFUU1mZ8vtPMigtUz5/eDgd4xr7OyRjTAMSGRZM7pFCn6zb+h5W0xOz1/PVpjzuH5ZsX/7GmBoXGRbC8WLfHAFYAqiGhRtyeHP5Lkb3bMUvLu/i73CMMQ1QZFgwxwtLfbJujxKAiIwWkU0ikikiUypYLiLyV2f5WhHp78xvIyKLRGSDiGSIyMNudZ4UkT0istp5jK253fK9zNwCJr6RTqeWjXnu5j52ktcY4xORYcEcL/JNAqjyHICIBANTgcuALGCFiMxR1R/cio0BOjuPwcBLznMJ8AtVXSki0UC6iCxwq/uiqj5Xc7tTO1SVx95fiyD8696BNA63UynGGN9oFBbCieJSysq0xruQe3IEMAjIVNVtqloEzALGlSszDnhNXZYBTUQkQVWzVXUlgKoWABuAxBqM3y++zdzPql2HePjSziQ1jfR3OMaYBizSuWD0ZEnNHwV4kgASAffxSLM4+0u8yjIi0h7oByx3mz3ZaTKaISJNPYzZ72Z8u534mHDuH5bs71CMMQ3cqQRwzAfnATxJABUdc5S/Q0GlZUSkMfAB8IiqHnFmvwR0BPoC2cDzFW5cJFVE0kQkLS/Pd8OiempLTgGLNuVxVe/WNm6/McbnIsNcTczHCmu+J5AnCSALaOM2nQTs9bSMiITi+vJ/U1U/PFVAVXNUtVRVy4CXcTU1nUVVp6tqiqqmxMXV/GBI3jhWWMKN05YSGiw8MNx+/RtjfO/UOcZjRf5JACuAziKSLCJhwHhgTrkyc4C7nN5AQ4DDqpotrq4x/wQ2qOoL7hVEJMFt8jpg/XnvRS2Z8e12Dp8o5tkbe5MQa1f5GmN8LzzE9TV9amyxmlRl9xVVLRGRycB8IBiYoaoZIjLRWT4NmAuMBTKB48C9TvWLgDuBdSKy2pn3a1WdCzwrIn1xNRXtACbU2F75wLz12Ty/YDOX94jnun5J/g7HGBMgwpwEUFxa8/cG9qj/ovOFPbfcvGlurxWYVEG9b6n4/ACqeqdXkfrR3kMnmPjGSgB+69ylxxhjasOpm0X54gjArgSuwvGiEsZPXwbAq/cOtG6fxpha1Tg8hA5xUYQG1/zFpnYFUxW+3pzHrvzjPD62O6O6tvR3OMaYANOjdQz//sXFPlm3HQFU4b20LBqHh3DPRe39HYoxxtSogD8C2H+0kPfSsigsKeXavom0bxEFuIZ5fvS9NSzcmMtPLu5oN203xjQ4AZ0APlubzaS3Vp6e/vOXWwBXm9tR56KLmwYk8fPLbKRPY0zDE5AJYM6avTSPCuP99N0VLj/15X/P0Pb87uoeNtKnMaZBCrgEcLSwhIfeXnV6+tLuLXnl7oEUl5Zx5EQxzRuH886KXfRIiKVXUqwfIzXGGN8KuATwxrKdZ0xf09c1Zl1ocBDNG4cDcMvAtrUelzHG1LaASgArduTzzOcbaRoZypW9ExjWqQWjL0iouqIxxjRAAZUAbpq2FIBnbujNFT1b+TkaY4zxr4Dp23joeNHp15f3iPdjJMYYUzcETALYuK8AgJn3DbJePcYYQwAlgLyCQgBax0b4ORJjjKkbAiYBHDpRDEBsZKifIzHGmLohYBLAkVMJoJElAGOMgQBKALNX7wEgPMTu42uMMRBACWBzzlF/h2CMMXVKQCSAw8ddzT+Dkpv5ORJjjKk7AiIBZOa5uoBOHNnBz5EYY0zd4VECEJHRIrJJRDJFZEoFy0VE/uosXysi/auqKyLNRGSBiGxxnpvWzC6d7Y/zNgHQMa6xrzZhjDH1TpUJQESCganAGKAHcKuIlL8z+higs/NIBV7yoO4UYKGqdgYWOtM+0TvRNaqn3c/XGGP+w5MjgEFApqpuU9UiYBYwrlyZccBr6rIMaCIiCVXUHQfMdF7PBK6t5r6c0+NXdmfLH8YQHGRXABtjzCmeDAaXCLjfOSULGOxBmcQq6sarajaAqmaLSIV3XBeRVFxHFQBHRWSTBzFXpAWw/zzr+pLF5R2Lyzt1NS6ou7E1xLjaVTTTkwRQ0c9m9bCMJ3UrparTgene1KmIiKSpakp111PTLC7vWFzeqatxQd2NLZDi8qQJKAto4zadBOz1sExldXOcZiKc51zPwzbGGFNdniSAFUBnEUkWkTBgPDCnXJk5wF1Ob6AhwGGneaeyunOAu53XdwOzq7kvxhhjvFBlE5CqlojIZGA+EAzMUNUMEZnoLJ8GzAXGApnAceDeyuo6q34GeFdE7gd2ATfV6J6drdrNSD5icXnH4vJOXY0L6m5sAROXqHrVJG+MMaaBCIgrgY0xxpzNEoAxxgSoBpUAROQmEckQkTIROWd3qdoensKT9YpIVxFZ7fY4IiKPOMueFJE9bsvG1lZcTrkdIrLO2Xaat/V9EZeItBGRRSKywfmbP+y2rEbfL18MhVITPIjrdieetSKyRET6uC2r8G9aS3FdLCKH3f4+v/W0ro/jeswtpvUiUioizZxlvny/ZohIroisP8dy332+VLXBPIDuQFfgKyDlHGWCga1AByAMWAP0cJY9C0xxXk8B/lhDcXm1XifGfUA7Z/pJ4FEfvF8exQXsAFpUd79qMi4gAejvvI4GNrv9HWvs/ars8+JWZizwOa7rXoYAyz2t6+O4hgJNnddjTsVV2d+0luK6GPj0fOr6Mq5y5a8G/u3r98tZ9wigP7D+HMt99vlqUEcAqrpBVau6Utgfw1N4u95LgK2qurOGtn8u1d1fv71fqpqtqiud1wXABlxXntc0Xw2F4vO4VHWJqh50Jpfhug7H16qzz359v8q5FXi7hrZdKVX9GsivpIjPPl8NKgF46FzDVkC54SmACoenOA/ernc8Z3/4JjuHfzNqqqnFi7gU+EJE0sU1NIe39X0VFwAi0h7oByx3m11T71dln5eqynhS15dxubsf16/IU871N62tuC4UkTUi8rmI9PSyri/jQkQigdHAB26zffV+ecJnny9PhoKoU0TkS6BVBYseV1VPLiar9vAUFa60kri8XE8YcA3wK7fZLwFP4YrzKeB54L5ajOsiVd0rrvGaFojIRudXy3mrwferMa5/1EdU9Ygz+7zfr4o2UcG8WhsKpRIer1tERuFKAMPcZtf439SLuFbiat486pyf+RjXSMJ14v3C1fzznaq6/yr31fvlCZ99vupdAlDVS6u5iiqHp1DX4HReDU9RWVwi4s16xwArVTXHbd2nX4vIy8CntRmXqu51nnNF5CNch55f4+f3S0RCcX35v6mqH7qt+7zfrwpUZyiUMA/q+jIuRKQ38AowRlUPnJpfyd/U53G5JWpUda6I/ENEWnhS15dxuTnrCNyH75cnfPb5CsQmIH8MT+HNes9qe3S+BE+5Dqiwt4Av4hKRKBGJPvUauNxt+357v0REgH8CG1T1hXLLavL98tVQKNVV5bpFpC3wIXCnqm52m1/Z37Q24mrl/P0QkUG4vocOeFLXl3E58cQCI3H7zPn4/fKE7z5fvjir7a8Hrn/2LKAQyAHmO/NbA3Pdyo3F1WtkK66mo1Pzm+O6Oc0W57lZDcVV4XoriCsS1z9CbLn6rwPrgLXOHzihtuLC1cNgjfPIqCvvF67mDHXek9XOY6wv3q+KPi/ARGCi81pw3fhoq7PdlMrq1uDnvaq4XgEOur0/aVX9TWsprsnOdtfgOjk9tC68X870PcCscvV8/X69DWQDxbi+v+6vrc+XDQVhjDEBKhCbgIwxxmAJwBhjApYlAGOMCVCWAIwxJkBZAjDGmABlCcAYYwKUJQBjjAlQ/w+hC7ihQgP9JAAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "np.savez(\"./output_icecube_background_count.npz\", x = sweep_dec, y = B_i)\n",
    "\n",
    "plt.figure()\n",
    "plt.plot(np.sin(sweep_dec), B_i)\n",
    "plt.ylim(0.0, 0.2)\n",
    "plt.legend()\n",
    "\n",
    "# make a figure of the data sin dec, just a raw plot\n",
    "plt.figure()\n",
    "plt.title(\"3 Year IceCube Data, $\\sin(\\delta)$, N=\"+str(len(data_dec))+\" Events\")\n",
    "plt.hist(data_sin_dec, range=(-1, 1), bins=50)\n",
    "plt.xlabel(\"$\\sin(\\delta)$\")\n",
    "plt.xlim(-1.0, 1.0)\n",
    "plt.plot([np.sin(rad_ll), np.sin(rad_ll)], [0.0, 10000.0], color=\"red\")\n",
    "plt.plot([np.sin(rad_ul), np.sin(rad_ul)], [0.0, 10000.0], color=\"red\")\n",
    "\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
