import React from 'react';

import { ValueOf } from 'shared/types';
import { ShirtSize } from 'themes';

import css from './Grid.module.scss';

export const GridMode = {
  AutoFill: 'auto-fill', // will squeeze as many items into a given space and minimum size
  AutoFit: 'auto-fit', // auto-fill but also stretch to fit the entire available space.
  ScrollableRow: 'scrollableRow',
} as const;

export type GridMode = ValueOf<typeof GridMode>;

interface Props {
  border?: boolean;
  children: React.ReactNode;
  count?: number;
  gap?: ShirtSize;
  minItemWidth?: number;
  mode?: GridMode | number;
}

const sizeMap = {
  [ShirtSize.Small]: '4px',
  [ShirtSize.Medium]: '8px',
  [ShirtSize.Large]: '16px',
};

const Grid: React.FC<Props> = ({
  border,
  gap = ShirtSize.Medium,
  minItemWidth = 240,
  mode = GridMode.AutoFit,
  children,
  count,
}: Props) => {
  const style = {
    gridGap: `calc(${sizeMap[gap]} + var(--theme-density) * 1px)`,
    gridTemplateColumns: '',
  };
  const classes = [css.base];

  if (border) classes.push(css.border);
  if (mode === GridMode.AutoFill || GridMode.AutoFit) {
    style.gridTemplateColumns = `repeat(${mode}, minmax(${minItemWidth}px, 1fr))`;
  }
  if (mode === GridMode.ScrollableRow) {
    classes.push(css.row);
    style.gridTemplateColumns = `repeat(${count}, minmax(${minItemWidth}px, ${minItemWidth}px))`;
  }
  return (
    <div className={classes.join(' ')} style={style}>
      {children}
    </div>
  );
};

export default Grid;
