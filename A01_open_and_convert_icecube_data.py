{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading filename: ./events/IC86_VII_exp.txt\n",
      "Loading filename: ./events/IC59_exp.txt\n",
      "Loading filename: ./events/IC86_I_exp.txt\n",
      "Loading filename: ./events/IC86_V_exp.txt\n",
      "Loading filename: ./events/IC79_exp.txt\n",
      "Loading filename: ./events/IC86_II_exp.txt\n",
      "Loading filename: ./events/IC40_exp.txt\n",
      "Loading filename: ./events/IC86_III_exp.txt\n",
      "Loading filename: ./events/IC86_VI_exp.txt\n",
      "Loading filename: ./events/IC86_IV_exp.txt\n"
     ]
    }
   ],
   "source": [
    "import glob\n",
    "import numpy as np\n",
    "\n",
    "data_files = glob.glob(\"./events/IC*.txt\")\n",
    "data_file_year = np.array([])\n",
    "data_day = np.array([])\n",
    "data_sigmas = np.array([])\n",
    "data_ra = np.array([])\n",
    "data_dec = np.array([])\n",
    "\n",
    "for data_file_name in data_files:\n",
    "    print(\"Loading filename: %s\" % data_file_name)\n",
    "    f = open(data_file_name)\n",
    "\n",
    "    year = int(2011)#data_file_name.split(\"-\")[-2])\n",
    "\n",
    "    data = np.loadtxt(data_file_name, dtype='float')\n",
    "    data_file_year = np.append(data_file_year,\n",
    "                               year * np.ones(len(data), dtype='int'))                                              \n",
    "    data_day = np.append(data_day, data[:, 0])\n",
    "    data_sigmas = np.append(data_sigmas, data[:, 2])\n",
    "    data_ra = np.append(data_ra, data[:, 3])\n",
    "    data_dec = np.append(data_dec, data[:, 4])\n",
    "\n",
    "np.savez(\"output_icecube_data10yr.npz\",\n",
    "         data_day=data_day,\n",
    "         data_sigmas=data_sigmas,\n",
    "         data_ra=data_ra,\n",
    "         data_dec=data_dec,\n",
    "         data_file_year=data_file_year)"
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