
from enum import Enum

from fava.model.model import Model
from fava.mesh import FlashAMR, FlashParticles

class FileType(Enum):
    CHK = 0
    PLT = 1
    PRT = 2
    CHK_PRT = 3
    PLT_PRT = 4


class FLASH(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.chk_files = {
            "by number": {int(str(p).split("hdf5_chk_")[-1]): p for p in self._filter_files("*hdf5_chk_*")},
            "by index": {i: p for i,p in enumerate(self._filter_files("*hdf5_chk_*"))}
        }
        self.plt_files = {
            "by number": {int(str(p).split("hdf5_plt_cnt_")[-1]): p for p in self._filter_files("*hdf5_plt_cnt_*")},
            "by index": {i: p for i,p in enumerate(self._filter_files("*hdf5_plt_cnt_*"))}
        }
        self.prt_files = {
            "by number": {int(str(p).split("hdf5_part_")[-1]): p for p in self._filter_files("*hdf5_part_*")},
            "by index": {i: p for i,p in enumerate(self._filter_files("*hdf5_part_*"))}
        }


    def nfiles(self, *args, **kwargs) -> int:
        file_type = kwargs.get("file_type", FileType.CHK)
        ftype_ = file_type if isinstance(file_type, FileType) else FileType[file_type.upper()]

        match ftype_:
            case FileType.CHK:
                n = len(self.chk_files["by index"].keys())
            case FileType.PLT:
                n = len(self.plt_files["by index"].keys())
            case FileType.PRT:
                n = len(self.prt_files["by index"].keys())
        return n

    def load(self, file_index: int = 0, file_number: int=None, file_type: FileType|str=FileType.CHK, fields = None, *args, **kwargs):

        file_ = None
        part_file_ = None

        ftype_ = file_type if isinstance(file_type, FileType) else FileType[file_type.upper()]

        fkey = "by index" if file_number is None else "by number"
        nkey = file_index if file_number is None else file_number

        self.mesh = None
        self.particles = None

        match ftype_:

            case FileType.CHK:
                assert nkey in self.chk_files[fkey]
                file_ = self.chk_files[fkey][nkey]
                self.mesh = FlashAMR(file_)
                self.mesh.load(*args, **kwargs)

            case FileType.PLT:
                assert nkey in self.plt_files[fkey]
                file_ = self.plt_files[fkey][nkey]
                self.mesh = FlashAMR(file_)
                self.mesh.load(*args, **kwargs)

            case FileType.PRT:
                assert nkey in self.prt_files[fkey]
                file_ = self.prt_files[fkey][nkey]
                self.particles = FlashParticles(file_)
                self.particles._load_particles(*args, **kwargs)

            case FileType.CHK_PRT:
                assert nkey in self.chk_files[fkey]
                file_ = self.chk_files[fkey][nkey]
                self.mesh = FlashAMR(file_)
                self.mesh.load(*args, **kwargs)
                self.particles = FlashParticles(file_)
                self.particles._load_particles(*args, **kwargs)

            case FileType.PLT_PRT:
                assert nkey in self.plt_files[fkey]
                file_ = self.plt_files[fkey][nkey]

                self.mesh = FlashAMR(file_)
                self.mesh.load(*args, **kwargs)

                assert nkey in self.prt_files[fkey]
                pfile_ = self.prt_files[fkey][nkey]

                self.particles = FlashParticles(pfile_)
                self.particles._load_particles(*args, **kwargs)

        