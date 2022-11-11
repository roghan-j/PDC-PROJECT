# mpirun -np 5 python D2N2Accel.py
from mpi4py import MPI

comm= MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


if rank==0:
    print("master working")

elif rank==1:
    print("Worker1 working")

elif rank==2:
    print("Worker2 working")

elif rank==3:
    print("Worker3 working")

elif rank==4:
    print("Worker4 working")
    
    