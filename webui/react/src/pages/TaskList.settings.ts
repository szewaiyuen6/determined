import { array, boolean, literal, number, string, undefined as undefinedType, union } from 'io-ts';

import { InteractiveTableSettings } from 'components/Table/InteractiveTable';
import { MINIMUM_PAGE_SIZE } from 'components/Table/Table';
import { SettingsConfig } from 'hooks/useSettings';
import { CommandState, CommandType } from 'types';

export type TaskColumnName =
  | 'action'
  | 'id'
  | 'startTime'
  | 'state'
  | 'name'
  | 'type'
  | 'resourcePool'
  | 'user';

export const DEFAULT_COLUMNS: TaskColumnName[] = [
  'id',
  'type',
  'name',
  'startTime',
  'state',
  'resourcePool',
  'user',
];

export const DEFAULT_COLUMN_WIDTHS: Record<TaskColumnName, number> = {
  action: 46,
  id: 100,
  name: 150,
  resourcePool: 128,
  startTime: 117,
  state: 106,
  type: 85,
  user: 85,
};

export const ALL_SORTKEY = [
  'id',
  'name',
  'resourcePool',
  'startTime',
  'state',
  'type',
  'user',
] as const;

type SORTKEYTuple = typeof ALL_SORTKEY;

export type SORTKEY = SORTKEYTuple[number];

export const isOfSortKey = (sortKey: React.Key): sortKey is SORTKEY => {
  return !!ALL_SORTKEY.find((d) => d === String(sortKey));
};

export interface Settings extends InteractiveTableSettings {
  columns: TaskColumnName[];
  row?: string[];
  search?: string;
  sortKey: SORTKEY;
  state?: CommandState[];
  type?: CommandType[];
  user?: string[];
}

const config: SettingsConfig<Settings> = {
  settings: {
    columns: {
      defaultValue: DEFAULT_COLUMNS,
      skipUrlEncoding: true,
      storageKey: 'columns',
      type: array(
        union([
          literal('action'),
          literal('id'),
          literal('startTime'),
          literal('state'),
          literal('name'),
          literal('type'),
          literal('resourcePool'),
          literal('user'),
        ]),
      ),
    },
    columnWidths: {
      defaultValue: DEFAULT_COLUMNS.map((col: TaskColumnName) => DEFAULT_COLUMN_WIDTHS[col]),
      skipUrlEncoding: true,
      storageKey: 'columnWidths',
      type: array(number),
    },
    row: {
      defaultValue: undefined,
      storageKey: 'row',
      type: union([undefinedType, array(string)]),
    },
    search: {
      defaultValue: undefined,
      storageKey: 'search',
      type: union([undefinedType, string]),
    },
    sortDesc: {
      defaultValue: true,
      storageKey: 'sortDesc',
      type: boolean,
    },
    sortKey: {
      defaultValue: 'startTime',
      storageKey: 'sortKey',
      type: union([
        literal('id'),
        literal('name'),
        literal('resourcePool'),
        literal('startTime'),
        literal('state'),
        literal('type'),
        literal('user'),
      ]),
    },
    state: {
      defaultValue: undefined,
      storageKey: 'state',
      type: union([
        undefinedType,
        array(
          union([
            literal(CommandState.Pulling),
            literal(CommandState.Queued),
            literal(CommandState.Running),
            literal(CommandState.Starting),
            literal(CommandState.Terminated),
            literal(CommandState.Terminating),
            literal(CommandState.Waiting),
          ]),
        ),
      ]),
    },
    tableLimit: {
      defaultValue: MINIMUM_PAGE_SIZE,
      storageKey: 'tableLimit',
      type: number,
    },
    tableOffset: {
      defaultValue: 0,
      storageKey: 'tableOffset',
      type: number,
    },
    type: {
      defaultValue: undefined,
      storageKey: 'type',
      type: union([
        undefinedType,
        array(
          union([
            literal(CommandType.Command),
            literal(CommandType.JupyterLab),
            literal(CommandType.Shell),
            literal(CommandType.TensorBoard),
          ]),
        ),
      ]),
    },
    user: {
      defaultValue: undefined,
      storageKey: 'user',
      type: union([undefinedType, array(string)]),
    },
  },
  storagePath: 'task-list',
};

export default config;
