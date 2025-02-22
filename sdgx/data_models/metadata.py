from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Literal, Set, Union
from tqdm import autonotebook as tqdm

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from sdgx.data_loader import DataLoader
from sdgx.data_models.inspectors.base import RelationshipInspector
from sdgx.data_models.inspectors.manager import InspectorManager
from sdgx.exceptions import MetadataInitError, MetadataInvalidError
from sdgx.models.components.utils import StrValuedBaseEnum
from sdgx.utils import logger


class CategoricalEncoderType(StrValuedBaseEnum):
    ONEHOT = "onehot"
    LABEL = "label"
    FREQUENCY = "frequency"

class ContinuousEncoderType(StrValuedBaseEnum):
    GMM = "gmm"
    NONE = "none"

class Metadata(BaseModel):
    """
    Metadata is mainly used to describe the data types of all columns in a single data table.

    For each column, there should be an instance of the Data Type object.

    .. Note::

        Use ``get``, ``set``, ``add``, ``delete`` to update tags in the metadata. And use `query` for querying a column for its tags.

    Args:
        primary_keys(List[str]): The primary key, a field used to uniquely identify each row in the table.
        The primary key of each row must be unique and not empty.

        column_list(list[str]): list of the comlumn name in the table, other columns lists are used to store column information.
    """

    primary_keys: Set[str] = set()
    """
    primary_keys is used to store single primary key or composite primary key
    """

    column_list: List[str] = Field(default_factory=list, title="The List of Column Names")

    """"
    column_list is the actual value of self.column_list
    """

    @field_validator("column_list")
    @classmethod
    def check_column_list(cls, value) -> Any:
        # check if v has duplicate element
        if len(value) == len(set(value)):
            return value
        raise MetadataInitError("column_list has duplicate element!")

    column_inspect_level: Dict[str, int] = defaultdict(lambda: 10)
    """
    column_inspect_level is used to store every inspector's level, to specify the true type of each column.
    """

    pii_columns: Set[str] = set()
    """
    pii_columns is used to store all PII columns' name
    """

    # other columns lists are used to store column information
    # here are 6 basic data types
    id_columns: Set[str] = set()
    int_columns: Set[str] = set()
    float_columns: Set[str] = set()
    bool_columns: Set[str] = set()
    discrete_columns: Set[str] = set()
    datetime_columns: Set[str] = set()
    const_columns: Set[str] = set()
    datetime_format: Dict = defaultdict(str)
    numeric_format: Dict = defaultdict(list)

    continuous_encoder: Dict[str, ContinuousEncoderType] = defaultdict(str)
    # def columns encoder, even not matched categorical_threshold.
    categorical_encoder: Union[Dict[str, CategoricalEncoderType], None] = defaultdict(str)
    # if greater than categorical_threshold, encoder is the mapped.
    categorical_threshold: Union[Dict[int, CategoricalEncoderType], None] = None

    # version info
    version: str = "1.0"
    _extend: Dict[str, Set[str]] = defaultdict(set)
    """
    For extend information, use ``get`` and ``set``
    """

    def get_column_encoder_by_categorical_threshold(
        self, num_categories: int
    ) -> Union[CategoricalEncoderType, None]:
        encoder_type = None
        if self.categorical_threshold is None:
            return encoder_type

        for threshold in sorted(self.categorical_threshold.keys()):
            if num_categories > threshold:
                encoder_type = self.categorical_threshold[threshold]
            else:
                break
        return encoder_type

    def get_column_encoder_by_name(self, column_name) -> Union[CategoricalEncoderType, None]:
        encoder_type = None
        if self.categorical_encoder and column_name in self.categorical_encoder:
            encoder_type = self.categorical_encoder[column_name]
        return encoder_type

    @property
    def tag_fields(self) -> Iterable[str]:
        """
        Return all tag fields in this metadata.
        """

        return chain(
            (k for k in self.model_fields if k.endswith("_columns")),
            (k for k in self._extend.keys() if k.endswith("_columns")),
        )

    @property
    def format_fields(self) -> Iterable[str]:
        """
        Return all tag fields in this metadata.
        """

        return chain(
            (k for k in self.model_fields if k.endswith("_format")),
            (k for k in self._extend.keys() if k.endswith("_format")),
        )

    def __eq__(self, other):
        if not isinstance(other, Metadata):
            return super().__eq__(other)
        return (
            set(self.tag_fields) == set(other.tag_fields)
            and all(
                self.get(key) == other.get(key)
                for key in set(chain(self.tag_fields, other.tag_fields))
            )
            and all(
                self.get(key) == other.get(key)
                for key in set(chain(self.format_fields, other.format_fields))
            )
            and self.version == other.version
        )

    def query(self, field: str) -> Iterable[str]:
        """
        Query all tags of a field.

        Args:
            field(str): The field to query.

        Example:

            .. code-block:: python

                # Assume that user_id looks like 1,2,3,4
                m.query("user_id") == ["id_columns", "numeric_columns"]
        """
        return (k for k in self.tag_fields if field in self.get(k))

    def get(self, key: str) -> Set[str]:
        """
        Get all tags by key.

        Args:
            key(str): The key to get.

        Example:

            .. code-block:: python

                # Get all id columns
                m.get("id_columns") == {"user_id", "ticket_id"}
        """

        if key == "_extend":
            raise MetadataInitError("Cannot get _extend directly")

        return getattr(self, key) if key in self.model_fields else self._extend[key]

    def set(self, key: str, value: Any):
        """
        Set tags, will convert value to set if value is not a set.

        Args:
            key(str): The key to set.
            value(Any): The value to set.

        Example:

            .. code-block:: python

                # Set all id columns
                m.set("id_columns", {"user_id", "ticket_id"})
        """

        if key == "_extend":
            raise MetadataInitError("Cannot set _extend directly")

        old_value = self.get(key)
        if (
            key in self.model_fields
            and key not in self.tag_fields
            and key not in self.format_fields
        ):
            raise MetadataInitError(
                f"Set {key} not in tag_fields, try set it directly as m.{key} = value"
            )

        if isinstance(old_value, Iterable) and not isinstance(old_value, str):
            # list, set, tuple...
            value = value if isinstance(value, Iterable) and not isinstance(value, str) else [value]
            try:
                value = type(old_value)(value)
            except TypeError as e:
                if type(old_value) == defaultdict:
                    value = dict(value)
                else:
                    raise e

        if key in self.model_fields:
            setattr(self, key, value)
        else:
            self._extend[key] = value

    def add(self, key: str, values: str | Iterable[str]):
        """
        Add tags.

        Args:
            key(str): The key to add.
            values(str | Iterable[str]): The value to add.

        Example:

            .. code-block:: python

                # Add all id columns
                m.add("id_columns", "user_id")
                m.add("id_columns", "ticket_id")
                # OR
                m.add("id_columns", ["user_id", "ticket_id"])
                # OR
                # add datetime format
                m.add('datetime_format',{"col_1": "%Y-%m-%d %H:%M:%S", "col_2": "%d %b %Y"})
        """

        values = (
            values if isinstance(values, Iterable) and not isinstance(values, str) else [values]
        )

        # dict support,  this prevents the value in the key-value pair from being discarded
        if isinstance(values, dict):
            # already in fields that contains dict
            if key in list(self.format_fields):
                self.get(key).update(values)

            # in extend
            if self._extend.get(key, None) is None:
                self._extend[key] = values
            else:
                self._extend[key].update(values)
            return

        for value in values:
            self.get(key).add(value)

    def delete(self, key: str, value: str):
        """
        Delete tags.

        Args:
            key(str): The key to delete.
            value(str): The value to delete.

        Example:

            .. code-block:: python

                # Delete misidentification id columns
                m.delete("id_columns", "not_an_id_columns")

        """
        try:
            self.get(key).remove(value)
        except KeyError:
            pass

    def update(self, attributes: dict[str, Any]):
        """
        Update tags.
        """
        for k, v in attributes.items():
            self.add(k, v)

        return self

    @classmethod
    def from_dataloader(
        cls,
        dataloader: DataLoader,
        max_chunk: int = 10,
        primary_keys: Set[str] = None,
        include_inspectors: Iterable[str] | None = None,
        exclude_inspectors: Iterable[str] | None = None,
        inspector_init_kwargs: dict[str, Any] | None = None,
        check: bool = False,
    ) -> "Metadata":
        """Initialize a metadata from DataLoader and Inspectors

        Args:
            dataloader(DataLoader): the input DataLoader.
            max_chunk(int): max chunk count.
            primary_keys(list[str]): primary keys, see :class:`~sdgx.data_models.metadata.Metadata` for more details.
            include_inspectors(list[str]): data type inspectors used in this metadata (table).
            exclude_inspectors(list[str]): data type inspectors NOT used in this metadata (table).
            inspector_init_kwargs(dict): inspector args.
        """
        logger.info("Initing metadata inspectors")
        im = InspectorManager()
        exclude_inspectors = exclude_inspectors or []
        exclude_inspectors.extend(
            name
            for name, inspector_type in im.registed_inspectors.items()
            if issubclass(inspector_type, RelationshipInspector)
        )

        inspectors = im.init_inspcetors(
            include_inspectors, exclude_inspectors, **(inspector_init_kwargs or {})
        )
        logger.info(f"Using metadata inspectors: {[i.__class__.__name__ for i in inspectors]}")
        # set all inspectors not ready
        for inspector in inspectors:
            inspector.ready = False
        # TODO TODO 进度条不显示？
        logger.info("Fitting metadata inspectors...")
        for i, chunk in tqdm.tqdm(enumerate(dataloader.iter()), desc="Fitting metadata ", delay=3):
            for inspector in tqdm.tqdm(inspectors, desc="Fitting metadata inspectors"):
                if not inspector.ready:
                    inspector.fit(chunk)
            if all(i.ready for i in inspectors) or i > max_chunk:
                break

        if primary_keys is None:
            primary_keys = set()

        metadata = Metadata(primary_keys=primary_keys, column_list=dataloader.columns())
        logger.info("Inspecting metadata...")
        for inspector in tqdm.tqdm(inspectors, desc="Metadata inspectors inspecting", delay=3):
            
            inspect_res = inspector.inspect()
            # update column type
            metadata.update(inspect_res)
            # update pii column
            if inspector.pii:
                for each_key in inspect_res:
                    metadata.update({"pii_columns": inspect_res[each_key]})
            # update inspect level
            for each_key in inspect_res:
                if "columns" in each_key:
                    metadata.column_inspect_level[each_key] = inspector.inspect_level
            logger.debug(f"Inspected by {inspector.__class__}: {inspect_res}")

        if not primary_keys:
            metadata.update_primary_key(metadata.id_columns)

        all_dtype_columns = metadata.get_all_data_type_columns()
        missing_cols = set(metadata.column_list) - set(all_dtype_columns)
        if missing_cols:
            logger.warning(f"Some columns' data type can't be specified by Inspector automatically: {missing_cols}\nHere are some infomation to decide manually:\n{dataloader.dtypes().loc[missing_cols]}")
        
        if check:
            metadata.check()
        return metadata

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        include_inspectors: list[str] | None = None,
        exclude_inspectors: list[str] | None = None,
        inspector_init_kwargs: dict[str, Any] | None = None,
        check: bool = False,
    ) -> "Metadata":
        """Initialize a metadata from DataFrame and Inspectors

        Args:
            df(pd.DataFrame): the input DataFrame.
            include_inspectors(list[str]): data type inspectors used in this metadata (table).
            exclude_inspectors(list[str]): data type inspectors NOT used in this metadata (table).
            inspector_init_kwargs(dict): inspector args.
        """

        im = InspectorManager()
        exclude_inspectors = exclude_inspectors or []
        exclude_inspectors.extend(
            name
            for name, inspector_type in im.registed_inspectors.items()
            if issubclass(inspector_type, RelationshipInspector)
        )

        inspectors = im.init_inspcetors(
            include_inspectors, exclude_inspectors, **(inspector_init_kwargs or {})
        )
        for inspector in inspectors:
            inspector.fit(df)

        metadata = Metadata(primary_keys=[df.columns[0]], column_list=df.columns)
        for inspector in inspectors:
            inspect_res = inspector.inspect()
            # update column type
            metadata.update(inspect_res)
            # update pii column
            if inspector.pii:
                for each_key in inspect_res:
                    metadata.update({"pii_columns": inspect_res[each_key]})
            # update inspect level
            for each_key in inspect_res:
                if "columns" in each_key:
                    metadata.column_inspect_level[each_key] = inspector.inspect_level
        
        all_dtype_columns = metadata.get_all_data_type_columns()
        missing_cols = set(metadata.column_list) - set(all_dtype_columns)
        if missing_cols:
            logger.warning(f"Some columns' data type can't be specified by Inspector automatically: {missing_cols}")
        if check:
            metadata.check()
        return metadata

    def _dump_json(self) -> str:
        return self.model_dump_json(indent=4)

    def save(self, path: str | Path):
        """
        Save metadata to json file.
        """

        with path.open("w") as f:
            f.write(self._dump_json())

    @classmethod
    def loads(cls, attributes: Dict):
        attributes = attributes.copy()
        # attributes["primary_keys"] = set(attributes["primary_keys"])
        # categorical_encoder
        # continuous_encoder
        # categorical_threshold
        return Metadata(**attributes)

    @classmethod
    def load(cls, path: str | Path) -> "Metadata":
        """
        Load metadata from json file.
        """

        path = Path(path).expanduser().resolve()
        attributes = json.load(path.open("r"))
        version = attributes.get("version", None)
        if version:
            cls.upgrade(version, attributes)

        m = Metadata(**attributes)

        return m

    @classmethod
    def upgrade(cls, old_version: str, fields: dict[str, Any]) -> None:
        pass

    def check_single_primary_key(self, input_key: str):
        """Check whether a primary key in column_list and has ID data type.

        Args:
            input_key(str): the input primary_key str
        """

        if input_key not in self.column_list:
            raise MetadataInvalidError(f"Primary Key {input_key} not Exist in columns.")

    def get_all_data_type_columns(self):
        """Get all column names from `self.xxx_columns`.

        All Lists with the suffix _columns in model fields and extend fields need to be collected.
        All defined column names will be counted.

        Returns:
            all_dtype_cols(set): set of all column names.
        """
        all_dtype_cols = set()

        # search the model fields and extend fields
        for each_key in list(self.model_fields.keys()) + list(self._extend.keys()):
            if each_key.endswith("_columns"):
                column_names = self.get(each_key)
                all_dtype_cols = all_dtype_cols.union(set(column_names))
        return all_dtype_cols

    def check(self):
        """Checks column info.

        When passing as input to the next module, perform necessary checks, including:
            -Is the primary key correctly defined(in column list) and has ID data type.
            -Is there any missing definition of each column in table.
            -Are there any unknown columns that have been incorrectly updated.
        """
        # check primary key in column_list
        for each_key in self.primary_keys:
            self.check_single_primary_key(each_key)

        # for single primary key, it should has ID type
        if len(self.primary_keys) == 1 and list(self.primary_keys)[0] not in self.id_columns:
            raise MetadataInvalidError(f"Primary Key {self.primary_keys} should has ID DataType.")

        all_dtype_columns = self.get_all_data_type_columns()

        # check missing columns
        if set(self.column_list) - set(all_dtype_columns):
            raise MetadataInvalidError(
                f"Undefined data type for column {set(self.column_list) - set(all_dtype_columns)}."
            )

        # check unfamiliar columns in dtypes
        if set(all_dtype_columns) - set(self.column_list):
            raise MetadataInvalidError(
                f"Found undefined column: {set(all_dtype_columns) - set(self.column_list)}."
            )

        if self.categorical_encoder is not None:
            for i in self.categorical_encoder.keys():
                if not isinstance(i, str) or i not in self.discrete_columns:
                    raise MetadataInvalidError(
                        f"categorical_encoder key {i} is invalid, it should be an str and is a discrete column name."
                    )
            if self.categorical_encoder.values() not in CategoricalEncoderType:
                raise MetadataInvalidError(
                    f"In categorical_encoder values, categorical encoder type invalid, now supports {list(CategoricalEncoderType)}."
                )

        if self.categorical_threshold is not None:
            for i in self.categorical_threshold.keys():
                if not isinstance(i, int) or i < 0:
                    raise MetadataInvalidError(
                        f"categorical threshold {i} is invalid, it should be an positive int."
                    )
            if self.categorical_threshold.values() not in CategoricalEncoderType:
                raise MetadataInvalidError(
                    f"In categorical_threshold values, categorical encoder type invalid, now supports {list(CategoricalEncoderType)}."
                )

        logger.debug("Metadata check succeed.")

    def update_primary_key(self, primary_keys: Iterable[str] | str):
        """Update the primary key of the table

        When update the primary key, the original primary key will be erased.

        Args:
            primary_keys(Iterable[str]): the primary keys of this table.
        """

        if not isinstance(primary_keys, Iterable) and not isinstance(primary_keys, str):
            raise MetadataInvalidError("Primary key should be Iterable or str.")
        primary_keys = set(primary_keys if isinstance(primary_keys, Iterable) else [primary_keys])

        if not primary_keys.issubset(set(self.column_list)):
            raise MetadataInvalidError("Primary key not exist in table columns.")

        self.primary_keys = primary_keys

        logger.info(f"Primary Key updated: {primary_keys}.")

    def dump(self):
        """Dump model dict, can be used in downstream process, like processor.

        Returns:
            dict: dumped dict.
        """
        model_dict = self.model_dump()
        model_dict["column_data_type"] = {}
        for each_col in self.column_list:
            model_dict["column_data_type"][each_col] = self.get_column_data_type(each_col)
        return model_dict

    def get_column_data_type(self, column_name: str):
        """Get the exact type of specific column.
        Args:
            column_name(str): The query colmun name.
        Returns:
            str: The data type query result.
        """
        if column_name not in self.column_list:
            raise MetadataInvalidError(f"Column {column_name}not exists in metadata.")
        current_type = None
        current_level = 0
        # find the dtype who has most high inspector level
        for each_key in list(self.model_fields.keys()) + list(self._extend.keys()):
            if (
                each_key != "pii_columns"
                and each_key.endswith("_columns")
                and column_name in self.get(each_key)
                and current_level < self.column_inspect_level[each_key]
            ):
                current_level = self.column_inspect_level[each_key]
                current_type = each_key
        if not current_type:
            raise MetadataInvalidError(f"Column {column_name} has no data type.")
        return current_type.split("_columns")[0]

    def get_column_pii(self, column_name: str):
        """Return if a column is a PII column.
        Args:
            column_name(str): The query colmun name.
        Returns:
            bool: The PII query result.
        """
        if column_name not in self.column_list:
            raise MetadataInvalidError(f"Column {column_name}not exists in metadata.")
        if column_name in self.pii_columns:
            return True
        return False

    def change_column_type(
        self, column_names: str | List[str], column_original_type: str, column_new_type: str
    ):
        """Change the type of column."""
        if not column_names:
            return
        if isinstance(column_names, str):
            column_names = [column_names]
        all_fields = list(self.tag_fields)
        original_type = f"{column_original_type}_columns"
        new_type = f"{column_new_type}_columns"
        if original_type not in all_fields:
            raise MetadataInvalidError(f"Column type {column_original_type} not exist in metadata.")
        if new_type not in all_fields:
            raise MetadataInvalidError(f"Column type {column_new_type} not exist in metadata.")
        type_columns = self.get(original_type)
        diff = set(column_names).difference(type_columns)
        if diff:
            raise MetadataInvalidError(f"Columns {column_names} not exist in {original_type}.")
        self.add(new_type, column_names)
        type_columns = type_columns.difference(column_names)
        self.set(original_type, type_columns)

    def remove_column(self, column_names: List[str] | str):
        """
        Remove a column from all columns type.
        Args:
            column_names: List[str]: To removed columns name list.
        """
        if not column_names:
            return
        if isinstance(column_names, str):
            column_names = [column_names]
        column_names = frozenset(column_names)
        inter = column_names.intersection(self.column_list)
        if not inter:
            raise MetadataInvalidError(f"Columns {inter} not exist in metadata.")

        def do_remove_columns(key, get=True, to_removes=column_names):
            obj = self
            if get:
                target = obj.get(key)
            else:
                target = getattr(obj, key)
            res = None
            if isinstance(target, list):
                res = [item for item in target if item not in to_removes]
            elif isinstance(target, dict):
                if key == "numeric_format":
                    obj.set(
                        key,
                        {k: {v2 for v2 in v if v2 not in to_removes} for k, v in target.items()},
                    )
                else:
                    res = {k: v for k, v in target.items() if k not in to_removes}
            elif isinstance(target, set):
                res = target.difference(to_removes)

            if res is not None:
                if get:
                    obj.set(key, res)
                else:
                    setattr(obj, key, res)

        to_remove_attribute = list(self.tag_fields)
        to_remove_attribute.extend(list(self.format_fields))
        for attr in to_remove_attribute:
            do_remove_columns(attr)
        for attr in ["column_list", "primary_keys", "categorical_encoder"]:
            do_remove_columns(attr, False)
        self.check()

    def to_json(self):
        def custom_encoder(obj):
            if isinstance(obj, set):
                return list(obj)  # Convert set to list
            if isinstance(obj, StrValuedBaseEnum):
                return obj.value
            raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')
        meta_dict = self.dump()
        return json.dumps(meta_dict, default=custom_encoder)

