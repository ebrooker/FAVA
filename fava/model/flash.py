from enum import Enum
from pathlib import Path

from fava.model.model import Model
from fava.mesh import FlashParticles, FlashUniform
from fava.mesh import FLASH as FlashAMR
from fava.util import mpi


class FileSubStem(Enum):
    CHK = "chk"
    PLT = "plt_cnt"
    PRT = "part"
    UNI = "uniform"
    ANL = "analysis"


class FileType(Enum):
    CHK = 0
    PLT = 1
    PRT = 2
    CHK_PRT = 3
    PLT_PRT = 4
    UNI = 5
    ANL = 6


class FLASH(Model):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.chk_files: dict[str, dict[int, Path]] = {
            "by number": {int(str(p).split("hdf5_chk_")[-1]): p for p in self._filter_files("*hdf5_chk_????")},
            "by index": {i: p for i, p in enumerate(self._filter_files("*hdf5_chk_????"))},
        }
        self.plt_files: dict[str, dict[int, Path]] = {
            "by number": {
                int(str(p).split("hdf5_plt_cnt_")[-1]): p for p in self._filter_files("*hdf5_plt_cnt_????")
            },
            "by index": {i: p for i, p in enumerate(self._filter_files("*hdf5_plt_cnt_????"))},
        }
        self.prt_files: dict[str, dict[int, Path]] = {
            "by number": {
                int(str(p).split("hdf5_part_")[-1]): p for p in self._filter_files("*hdf5_part_????")
            },
            "by index": {i: p for i, p in enumerate(self._filter_files("*hdf5_part_????"))},
        }

        self.uni_files: dict[str, dict[int, Path]] = {
            "by number": {
                int(str(p).split("hdf5_uniform_")[-1]): p for p in self._filter_files("*hdf5_uniform_????")
            },
            "by index": {i: p for i, p in enumerate(self._filter_files("*hdf5_uniform_????"))},
        }

        self.anl_files: dict[str, dict[int, Path]] = {
            "by number": {
                int(str(p).split("hdf5_analysis_")[-1]): p for p in self._filter_files("*hdf5_analysis_????")
            },
            "by index": {i: p for i, p in enumerate(self._filter_files("*hdf5_analysis_????"))},
        }

    def nfiles(self, *args, **kwargs) -> int:
        file_type = kwargs.get("file_type", FileType.CHK)
        ftype_: FileType = file_type if isinstance(file_type, FileType) else FileType[file_type.upper()]

        match ftype_:
            case FileType.CHK:
                n = len(self.chk_files["by index"].keys())
            case FileType.PLT:
                n = len(self.plt_files["by index"].keys())
            case FileType.PRT:
                n = len(self.prt_files["by index"].keys())
            case FileType.UNI:
                n = len(self.uni_files["by index"].keys())
            case FileType.ANL:
                n = len(self.anl_files["by index"].keys())
            case _:
                pass
        return n

    def load(
        self,
        file_index: int = 0,
        file_number: int = None,
        file_type: FileType | str = FileType.CHK,
        fields=None,
        *args,
        **kwargs,
    ):

        file_ = None
        part_file_ = None

        ftype_: FileType = file_type if isinstance(file_type, FileType) else FileType[file_type.upper()]

        fkey = "by index" if file_number is None else "by number"
        nkey = file_index if file_number is None else file_number

        self.mesh = None
        self.particles = None

        match ftype_:

            case FileType.CHK:
                assert nkey in self.chk_files[fkey]
                file_ = self.chk_files[fkey][nkey]
                self.mesh = FlashAMR(filename=file_)
                self.mesh.load(*args, **kwargs)

            case FileType.PLT:
                assert nkey in self.plt_files[fkey]
                file_ = self.plt_files[fkey][nkey]
                self.mesh = FlashAMR(filename=file_)
                self.mesh.load(*args, **kwargs)

            case FileType.PRT:
                assert nkey in self.prt_files[fkey]
                file_ = self.prt_files[fkey][nkey]
                self.particles = FlashParticles(filename=file_)
                self.particles._load_particles(*args, **kwargs)

            case FileType.CHK_PRT:
                assert nkey in self.chk_files[fkey]
                file_ = self.chk_files[fkey][nkey]
                self.mesh = FlashAMR(filename=file_)
                self.mesh.load(*args, **kwargs)
                self.particles = FlashParticles(filename=file_)
                self.particles._load_particles(*args, **kwargs)

            case FileType.PLT_PRT:
                assert nkey in self.plt_files[fkey]
                file_ = self.plt_files[fkey][nkey]

                self.mesh = FlashAMR(filename=file_)
                self.mesh.load(*args, **kwargs)

                assert nkey in self.prt_files[fkey]
                pfile_: Path = self.prt_files[fkey][nkey]

                self.particles = FlashParticles(filename=pfile_)
                self.particles._load_particles(*args, **kwargs)

            case FileType.UNI:
                assert nkey in self.uni_files[fkey]
                file_: Path = self.uni_files[fkey][nkey]
                if mpi.root:
                    print(file_)
                self.mesh = FlashUniform(filename=file_)
                self.mesh.load(*args, **kwargs)

    def convert_filename_type(self, current_filetype: FileType | str, new_filetype: FileType | str) -> str:
        if self.mesh is None:
            return

        curr_ftype_: FileType = (
            current_filetype if isinstance(current_filetype, FileType) else FileType[current_filetype.upper()]
        )
        new_ftype_: FileType = (
            new_filetype if isinstance(new_filetype, FileType) else FileType[new_filetype.upper()]
        )

        current_stem: str = self.mesh.filename.stem
        new_stem: str = current_stem.replace(
            FileSubStem[curr_ftype_.name].value, FileSubStem[new_ftype_.name].value
        )

        return self.mesh.filename.with_stem(new_stem)
