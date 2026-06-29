from pathlib import Path
from typing import Callable, Any
class DataComponent:
    def __init__(
            self,
            name: str,
            filename: str = "",
            relative_path:str = "",
            loader: Callable = None,
            saver: Callable = None
            ):
        
        self.name = name
        self.filename = filename
        self.relative_path = relative_path
        self.loader = loader
        self.saver = saver

    def full_path(self, base_dir: Path, **params) -> Path:
        relative_path_formatted = self.relative_path.format(**params)
        filename_formatted = self.filename.format(**params)
        return base_dir/relative_path_formatted/filename_formatted
    
    def exists(self, base_dir:Path, **params) -> bool:
        return self.full_path(base_dir, **params).exists()
    
    def load(self, base_dir:Path, **params):
        if self.loader is None:
            raise ValueError(f'No loader defined for {self.name}')
        return self.loader(self.full_path(base_dir, **params))
    
    def save(self, base_dir:Path, data:Any, **params):
        if self.saver is None:
            raise ValueError(f"No saver defined for {self.name}")
        self.full_path(base_dir, **params).parent.mkdir(exist_ok=True,parents=True)
        return self.saver(self.full_path(base_dir,**params), data)
    
    def clone_with_prefix(self, prefix: str, change_name = True) -> "DataComponent":
        return DataComponent(
            name=f"{prefix}_{self.name}" if change_name else self.name,
            filename=self.filename if self.filename else "",
            relative_path=self.relative_path + f"/{prefix}" if self.relative_path else "",
            loader=self.loader,
            saver=self.saver,
        )