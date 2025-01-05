from mpi4py import MPI


class FAVA_MPI:
    root: bool = False
    procs: int = 1
    id: int = 0

    def __init__(self) -> None:

        # Note: This may change in the future if we choose to manually
        # handle to instantiation of MPI Communicators, for now we use
        # the MPI.Intracomm baked in one COMM_WORLD
        self.comm: MPI.Intracomm = MPI.COMM_WORLD

        self.procs = self.comm.Get_size()
        self.id = self.comm.Get_rank()
        self.root = self.id == 0

        self.__windows = {}

    def __del__(self) -> None:
        if self.root:
            print(f"MPI ROOT: Cleaning up shared memory")
        self.clear_shared_memory()

    def print(self, after: bool = False, *args, **kwargs) -> None:
        if after:
            print(f"[MPI ID #{mpi.id} AFTER] ", *args, **kwargs)
        else:
            print(f"[MPI ID #{mpi.id} BEFOR] ", *args, **kwargs)

    def allocate(self, id: str, nbytes: int, itemsize: int) -> None | MPI.Win:

        if id not in self.__windows:
            _nbytes: int = 0
            if mpi.root:
                _nbytes += nbytes

            win: MPI.Win = MPI.Win.Allocate_shared(size=_nbytes, disp_unit=itemsize, comm=mpi.comm)
            self.__windows[id] = win

        else:
            win = self.__windows[id]

        return win

    def reallocate(self, id: str, nbytes: int, itemsize: int) -> None | MPI.Win:
        self.deallocate(id=id)
        return self.allocate(id=id, nbytes=nbytes, itemsize=itemsize)

    def deallocate(self, id: str) -> None:
        if id not in self.__windows:
            return
        win: MPI.Win = self.__windows[id]
        win.Fence()
        win.Free()
        self.__windows.pop(id)

    def clear_shared_memory(self) -> None:
        for id in list(self.__windows.keys()):
            self.deallocate(id=id)
        self.__windows: dict = {}

    def parallel_range(self, iterations: int) -> tuple[int, int]:
        extra: int = iterations % mpi.procs
        local_iterations: int = iterations // mpi.procs
        if mpi.id < extra:
            local_iterations += 1
            start: int = local_iterations * mpi.id
        else:
            start: int = extra * (local_iterations + 1) + (mpi.id - extra) * local_iterations

        return start, start + local_iterations


mpi = FAVA_MPI()
